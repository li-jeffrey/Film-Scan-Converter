import dataclasses
import logging
import os
import threading

import cv2
import numpy as np
import rawpy  # noqa: F401  (imported so a missing rawpy fails loudly at startup)
from PIL import Image

from conversion import decode, icc, pipeline
from conversion.params import (
    FILM_TYPE_BW,
    FILM_TYPE_CROP,
    FILM_TYPE_SLIDE,
    SETTINGS_VERSION,
    SPEC_DEFAULTS,
    from_photo,
    migrate_v1,
)
from conversion.print_curve import calculate_wb_shifts_from_log

logger = logging.getLogger(__name__)

ANALYSIS_LONG_EDGE = 2048


class RawProcessing:
    # This class defines a photo object that contains the image processing pipeline from raw to final export, including all processing functions and parameters
    default_parameters = dict(
        max_proxy_size = 2000, # Max dimension of height + width to increase processing speed (only for previews)
        histogram_plt_size = (1600, 2400, 3), # Dimensions of the histogram
        hist_bg_colour = (25, 25, 25), # sets the background colour of the histogram
        frame = 0, # adds white frame to final photo
        jpg_quality = 90, # from 0-100
        tiff_compression = 8, # defines the tiff compression algorithm
        dm_alg = 2, # demosaicing algorithm
        fbdd_nr = 0, # noise reduction on demosaic
        noise_thr = 0, # for RAW denoising
        median_filter_passes = 0, # can reduce colour artifacts during demosaicing
        ignore_border = (5, 5), # analysis border (% of the crop) excluded from metering
        ignore_neg_border = True, # ignores negative border space for metering
        output_colour_space = 'sRGB', # export colour space: sRGB / Adobe RGB
        input_encoding = 'Auto', # encoding of non-RAW 16-bit files without a profile: Auto / sRGB / Linear
        dust_threshold = 10,
        max_dust_area = 15,
        dust_iter = 5,
        picker_radius = 0.5,
        filetype = 'JPG',
        fit_aspect_ratio = 'Keep Original'
    )
    class_parameters = default_parameters.copy()
    advanced_attrs = [key for key in default_parameters.keys() if key not in ('filetype', 'frame', 'fit_aspect_ratio')] # list of keys for advanced settings, except for keys that should not be saved
    conversion_parameters = tuple(SPEC_DEFAULTS.keys()) # per-photo conversion controls (see conversion.params.SPECS)
    processing_parameters = ('dark_threshold','light_threshold','border_crop','flip','rotation','film_type','reject','base_detect','base_rgb','remove_dust') + conversion_parameters

    def __init__(self, file_directory, default_settings, global_settings, config_path):
        # file_directory: the name of the RAW file to be processed
        # Instance Variables
        self.processed = False # flag for whether the image has been processed yet
        self.proxy = False # Flag to keep track of whether or not proxies are being used
        self.FileReadError = False # flag for if the class could not read the RAW file
        self.active_processes = 0 # lets object know how many processes are currently processing the photo
        self.pick_wb = False # flag to pick white balance from a set of pixel coordinates
        self.file_directory = file_directory
        self.filename = os.path.basename(self.file_directory)
        self.colour_desc = None # RAW bayer colour description
        self.config_path = config_path
        self._init_caches()
        # initializing raw processing parameters
        try: # to read in the parameters from a saved file
            directory = os.path.join(self.config_path, f"{os.path.splitext(self.filename)[0]}.npy")
            params_dict = np.load(directory, allow_pickle=True).item()
        except Exception as e:# file does not exist
            logger.info(f'No saved settings for {self.filename}: {e}')
            for attr in self.processing_parameters:
                if attr in global_settings:
                    setattr(self, attr, global_settings[attr]) # Initializes every instance parameter based on default value
                else:
                    setattr(self, attr, 0) # otherwise set to 0
            self.use_global_settings = True # tells GUI class whether to overwrite current photo settings with global setting
        else: # import successful
            if params_dict.get('settings_version', 1) < SETTINGS_VERSION:
                params_dict = migrate_v1(params_dict) # old per-photo settings: keep geometry/film type, reset the conversion
            for attr in self.processing_parameters:
                if attr in params_dict:
                    setattr(self, attr, params_dict[attr]) # Initializes every instance parameter with imported parameters
                elif attr in default_settings:
                    setattr(self, attr, default_settings[attr]) # if parameter doesn't exist, use default
                else:
                    setattr(self, attr, 0) # otherwise set to 0

            self.use_global_settings = False # tells GUI class whether to overwrite current photo settings with global setting

    # ------------------------------------------------------------------ caches / pickling
    def _init_caches(self):
        self._lock = threading.Lock() # the GUI thread and the full-res preview thread both call process()
        self._grid = None
        self._grid_key = None
        self._analysis = None
        self._analysis_key = None
        self._norm = None
        self._norm_key = None
        self._img_space = None
        self._img_full_res = False
        self._lin_adobe = False

    _ARRAY_ATTRS = ('RAW_LIN', 'RAW_IMG', 'ANALYSIS_SRC', 'proxy_RAW_LIN', 'proxy_RAW_IMG', 'thresh', 'IMG', 'dust_mask', '_grid', '_norm')

    def __getstate__(self):
        # Batch export pickles photos into worker processes: drop the arrays and the lock, keep the small Analysis
        state = self.__dict__.copy()
        for attr in self._ARRAY_ATTRS + ('_lock',):
            state.pop(attr, None)
        state['_grid'] = None
        state['_norm'] = None
        state['_norm_key'] = None
        state['processed'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def _camera_gain(self):
        # As-shot WB as a linear gain: negatives/B&W only (slides fold it into the camera matrix)
        if self.film_type == FILM_TYPE_SLIDE or getattr(self, 'camera_wb', None) is None:
            return None
        return self.camera_wb

    def load(self, full_res=False):
        self.camera_wb = None
        self.rgb_xyz_matrix = None
        self.is_raw = False
        try:
            lin, meta = decode.decode_raw(self.file_directory, self.class_parameters, full_res)
            self.camera_wb, self.rgb_xyz_matrix, self.colour_desc = meta.camera_wb, meta.rgb_xyz_matrix, meta.colour_desc
            self.is_raw = True
        except Exception as _:
            try:
                to_adobe = (self.film_type == FILM_TYPE_SLIDE)
                loaded = decode.load_non_raw(self.file_directory, self.class_parameters['input_encoding'], to_adobe)
                lin = loaded.lin
                self._lin_adobe = loaded.in_adobe_primaries
            except Exception as e:
                logger.exception(f'Exception: {e}') # If fails again, set error attributes
                self.reject = True
                self.FileReadError = True
                return
        self._set_source(lin)

    def load_lightroom_tiff(self):
        # 16-bit Adobe RGB TIFF from Lightroom; linearized with its embedded profile. Export always embeds AdobeCompat.
        self.camera_wb = None
        self.rgb_xyz_matrix = None
        self.is_raw = False
        self.lightroom = True
        try:
            loaded = decode.load_lightroom_tiff(self.file_directory)
            self._lin_adobe = loaded.in_adobe_primaries
            self._set_source(loaded.lin)
        except Exception as e:
            logger.exception(f'Exception: {e}')
            self.reject = True
            self.FileReadError = True

    def _set_source(self, lin):
        # lin: linear uint16 RGB. Builds the display copy (RAW_IMG, BGR) and the metering source
        lin = np.ascontiguousarray(lin)
        wb_lin = decode.apply_wb(lin, self.camera_wb) # as-shot WB for display only
        gain = decode.display_gain(wb_lin)
        self._display_lut = decode.build_display_lut(gain)
        disp = decode.apply_display_lut(wb_lin, self._display_lut, bgr=True, bits16=True)
        if wb_lin is not lin:
            del wb_lin
        if self.is_raw:
            # RawPy sometimes misreads the RAW image size, this removes any extra black borders
            gray = cv2.cvtColor(cv2.convertScaleAbs(disp, alpha=(255.0/65535.0)), cv2.COLOR_BGR2GRAY)
            x, y, w, h = decode.black_border_box(gray)
            lin = np.ascontiguousarray(lin[y:y+h, x:x+w])
            disp = np.ascontiguousarray(disp[y:y+h, x:x+w])
        self.RAW_LIN = lin
        self.RAW_IMG = disp
        h, w = lin.shape[:2]
        scale = ANALYSIS_LONG_EDGE / max(h, w)
        if scale < 1:
            self.ANALYSIS_SRC = cv2.resize(lin, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            self.ANALYSIS_SRC = lin
        self._grid = self._analysis = self._norm = None
        self._grid_key = self._analysis_key = self._norm_key = None
        self.FileReadError = False
        self.memory_alloc = self.RAW_LIN.nbytes * 5 # estimation of memory requirements based on the size of the image

    def _render_image(self, output=None):
        # Returns a numpy array for the requested processing stage (export or preview source).
        if self.FileReadError:
            return
        match output:
            case 'RAW':
                img = self.rotate(self.RAW_IMG)
            case 'Threshold':
                img = self.rotate(self.thresh)
            case 'Contours':
                thresh_img = np.uint8(cv2.cvtColor(self.thresh, cv2.COLOR_GRAY2BGR) / 2)
                thresh_img[:,:,2] = 0

                indices = np.indices(thresh_img.shape[:2])
                zebra_width = int(np.max(indices) / 100)
                zebra = np.repeat((np.mod(indices[0] + indices[1], zebra_width * 2) > zebra_width)[:, :, np.newaxis], 3, axis=2)
                thresh_img = np.where(zebra, 0, thresh_img)
                img = cv2.addWeighted(cv2.convertScaleAbs(self.RAW_IMG, alpha=(255.0/65535.0)), 1, thresh_img, 0.2, 0)

                if self.rect is not None:
                    border_width = np.ceil((img.shape[0] + img.shape[1]) / 800)
                    if img.shape[0] > img.shape[1]:
                        x_crop = self.border_crop
                        y_crop = self.border_crop * img.shape[1] / img.shape[0]
                    else:
                        y_crop = self.border_crop
                        x_crop = self.border_crop * img.shape[0] / img.shape[1]
                    y, x = img.shape[0], img.shape[1]
                    rect = ((self.rect[0][0]*y, self.rect[0][1]*x), (self.rect[1][0]*y, self.rect[1][1]*x), self.rect[2])
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    extra_crop_box = self.shrink_box(box, x_crop, y_crop)

                    EQ_ignore_box = extra_crop_box.copy()
                    ignore_border = np.array(self.class_parameters['ignore_border'])
                    if self.border_crop < 0 and self.class_parameters['ignore_neg_border']:
                        EQ_ignore_box = self.shrink_box(EQ_ignore_box, -x_crop, -y_crop)
                    EQ_ignore_box = self.shrink_box(EQ_ignore_box, ignore_border[0], ignore_border[1])
                    EQ_ignore_poly = np.zeros_like(img)
                    cv2.fillPoly(EQ_ignore_poly, [extra_crop_box], (0,0,255))
                    cv2.fillPoly(EQ_ignore_poly, [EQ_ignore_box], (0,0,0))
                    img = np.where(np.dstack([np.sum(EQ_ignore_poly, (2)) == 0]*3), img, cv2.addWeighted(EQ_ignore_poly, 2, img, 0.8, 0))

                    cv2.drawContours(img,[box],0,(0,255,255), int(border_width * 0.75))
                    cv2.drawContours(img, self.largest_contour, -1, (0,255,255), int(border_width * .75))
                    cv2.drawContours(img,[extra_crop_box],0,(0,255,0), int(border_width))
                img = self.rotate(img)
            case 'Histogram':
                img = self.draw_histogram(self.IMG)
                img = np.flip(img, 0)
                mask = np.sum(img, 2) == 0
                img[mask] = np.array(self.class_parameters['hist_bg_colour'])
            case _:
                img = self.IMG
                if self.remove_dust:
                    img = self.fill_dust(img, self.dust_mask)
                img = self.add_frame(img)
                img = self.rotate(img)
        return img

    def get_export_img(self):
        # Full-resolution numpy image for disk export (16-bit, BGR or grayscale).
        return self._render_image()

    def get_preview_img(self, output=None):
        # 8-bit PIL image for GUI display (downscaled separately via resize_IMG).
        img = self._render_image(output)
        if img is None:
            return
        if img.dtype == np.uint16:
            img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        if img.ndim == 2:
            return Image.fromarray(img, mode='L')
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __str__(self):
        # returns file name when str() is called on photo
        return os.path.basename(self.file_directory)

    # ------------------------------------------------------------------ export
    def _export_space(self, lightroom=False):
        if self.film_type == FILM_TYPE_CROP:
            return 'display'
        if self.film_type == FILM_TYPE_BW:
            return 'Gray'
        return 'Adobe RGB' if lightroom else self.class_parameters['output_colour_space']

    def _ensure_space(self, space):
        # IMG is rendered in sRGB for previews; re-render when the export needs another space
        if getattr(self, '_img_space', None) != space or not hasattr(self, 'IMG'):
            self.process(self._img_full_res, skip_crop=True, for_export=True, out_space=None if space in ('display', 'Gray') else space)

    def _icc_for(self, space):
        if space == 'display':
            return None
        return icc.bundled_icc(space)

    def _write(self, filename, space):
        img = self.get_export_img()
        if img.ndim == 3:
            img = np.ascontiguousarray(img[:, :, ::-1]) # BGR -> RGB
        icc.write_image(
            filename, img, self.class_parameters['filetype'], self._icc_for(space),
            jpg_quality=self.class_parameters['jpg_quality'], tiff_compression=self.class_parameters['tiff_compression'],
        )

    def export(self, filename):
        # Saves final image to disk. filename: base path without extension
        space = self._export_space()
        self._ensure_space(space)
        self._write(f"{filename}.{self.class_parameters['filetype']}", space)

    def export_lightroom_tiff(self, filename):
        space = self._export_space(lightroom=True)
        self._ensure_space(space)
        img = self.get_export_img()
        if img.ndim == 3:
            img = np.ascontiguousarray(img[:, :, ::-1])
        import tifffile
        tifffile.imwrite(
            filename, img, photometric='minisblack' if img.ndim == 2 else 'rgb', compression='deflate',
            iccprofile=self._icc_for(space),
        )

    def save_settings(self):
        # saves the processing parameters to a file
        directory = os.path.join(self.config_path, f"{os.path.splitext(self.filename)[0]}.npy") # uses the same name as the input file
        params_dict = dict()
        for attr in self.processing_parameters:
            params_dict[attr] = getattr(self, attr)
        params_dict['settings_version'] = SETTINGS_VERSION
        np.save(directory, params_dict)

    # ------------------------------------------------------------------ processing
    def _params(self):
        params = from_photo(self)
        # Film base override only applies to colour negatives with a picked/imported base
        if params.base_rgb is not None and not (self.base_detect and self.film_type == 1):
            params = dataclasses.replace(params, base_rgb=None)
        return params

    def _crop_key(self):
        return (self.border_crop, self.dark_threshold, self.light_threshold, tuple(self.class_parameters['ignore_border']), self.class_parameters['ignore_neg_border'])

    def _ensure_analysis(self, params, cam_matrix):
        # Metering on the 2048 px analysis source (so preview and export agree), cached on the keys it depends on
        key = (params.analysis_key(), self._crop_key(), self.class_parameters['dm_alg'])
        with self._lock:
            if self._analysis is not None and self._analysis_key == key:
                return self._analysis
            gkey = (params.grid_key(), self._crop_key())
            if self._grid is None or self._grid_key != gkey:
                src = self.crop(self.ANALYSIS_SRC, self.rect, include_EQ_ignore=True)
                src = decode.apply_wb(src, self._camera_gain())
                self._grid = pipeline.build_grid(src, params, cam_matrix)
                self._grid_key = gkey
            self._analysis = pipeline.analyze(self._grid, params)
            self._analysis_key = key
            return self._analysis

    def process(self, full_res=False, recent_only=False, skip_crop=False, for_export=False, out_space=None):
        # Selection of appropriate film processing type
        # full_res uses the full resolution raw image for processing, otherwise use downsampled proxy
        # recent_only will not update final output if multiple, more recent processes are started before it finishes
        # skip_crop: skip calculating crop (faster if crop parameters have not changed)
        # for_export: render in the Advanced output colour space instead of sRGB
        if not hasattr(self, 'RAW_IMG'):
            self.load() # tries to load file if it hasn't already been done
        if self.FileReadError:
            return # Do not process if the file could not be read
        if not self.is_raw and not getattr(self, 'lightroom', False) and (self.film_type == FILM_TYPE_SLIDE) != self._lin_adobe:
            self.load() # non-RAW slides need Adobe-primaries input; reload when the film type crosses that line
            if self.FileReadError:
                return
            skip_crop = False
        self.active_processes += 1 # Used to keep track of the number of processes currently running

        try:
            if not skip_crop or not hasattr(self, 'thresh'):
                self.thresh, self.rect, self.largest_contour = self.find_optimal_crop()

                self.proxy = False
                img_size = self.RAW_IMG.shape[0] + self.RAW_IMG.shape[1]
                if (img_size > self.class_parameters['max_proxy_size']): # Checks if image is larger than the allowable size, if yes, then generate proxy images to speed up preview generation
                    # Downscales the image to a smaller size
                    scale_factor = self.class_parameters['max_proxy_size'] / img_size
                    x = int(self.RAW_IMG.shape[1] * scale_factor)
                    y = int(self.RAW_IMG.shape[0] * scale_factor)
                    self.proxy_RAW_IMG = cv2.resize(self.RAW_IMG, (x, y), interpolation=cv2.INTER_AREA)
                    self.proxy_RAW_LIN = cv2.resize(self.RAW_LIN, (x, y), interpolation=cv2.INTER_AREA)
                    self.proxy = True # Flag to tell the rest of the program that proxies are being used
                    self._norm = None

            # Uses a proxy to generate preview, when needed. During final export, will use full resolution
            use_proxy = self.proxy and not full_res
            disp = self.proxy_RAW_IMG if use_proxy else self.RAW_IMG
            lin = self.proxy_RAW_LIN if use_proxy else self.RAW_LIN

            dust_mask = self.find_dust(self.crop(disp, self.rect))

            if self.film_type == FILM_TYPE_CROP:
                img = self.crop(disp, self.rect) # camera-space display transform, no conversion
                space = 'display'
            else:
                params = self._params()
                cam_matrix = pipeline.camera_matrix(params, getattr(self, 'rgb_xyz_matrix', None), getattr(self, 'camera_wb', None))
                analysis = self._ensure_analysis(params, cam_matrix)
                lin = decode.apply_wb(self.crop(lin, self.rect), self._camera_gain()) # crop before converting: every step is per-pixel
                space = out_space or (params.output_colour_space if for_export else 'sRGB')
                norm = None
                if not full_res:
                    nkey = (params.norm_key(), self._crop_key())
                    with self._lock:
                        if self._norm is None or self._norm_key != nkey:
                            self._norm = pipeline.normalize(lin, params, analysis, cam_matrix)
                            self._norm_key = nkey
                        norm = self._norm
                img = pipeline.render(lin, params, analysis, space, norm=norm, cam_matrix=cam_matrix, bgr=True)
                if self.film_type == FILM_TYPE_BW:
                    space = 'Gray'
        finally:
            self.active_processes -= 1
        if recent_only and (self.active_processes > 0):
            return # skip displaying final image if it is not the last active process

        # Set all the global variables once processing is finished
        self.IMG = img
        self._img_space = space
        self._img_full_res = full_res
        self.dust_mask = dust_mask
        self.processed = True

    def effective_grade(self):
        # Read-out for the "Effective grade" label (None when not applicable)
        if self.film_type in (FILM_TYPE_CROP, FILM_TYPE_SLIDE) or self._analysis is None:
            return None
        try:
            return pipeline.effective_grade(self._params(), self._analysis)
        except Exception:
            return None

    def find_dust(self, img):
        # work in progress. Tries to detect dust particles of a maximum size, then returns threshold image of just dust
        y, x, _ = img.shape
        img_size = (x + y) / 2
        multiplier = img_size / 800
        max_dust_size = multiplier ** 2 * self.class_parameters['max_dust_area']
        kernel_size = max(round(multiplier) * 2 + 1,1)
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        x, y = (np.array(self.class_parameters['ignore_border']) / 100 * img.shape[:2][::-1]).astype(np.int32) # calculates the width of the border to ignore in pixels
        if x * y == 0:
            sample = np.s_[:]
        else:
            sample = np.s_[y:-y, x:-x]

        img8 = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        imgray = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY) # converts to b&w
        minimum = np.percentile(imgray[sample], 0.5)
        maximum = np.percentile(imgray[sample], 99.5)
        threshold = (maximum - minimum) * self.class_parameters['dust_threshold'] / 100 + minimum
        _, thresh = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY_INV)
        thresh_img = cv2.dilate(thresh,kernel,iterations = self.class_parameters['dust_iter'])
        thresh_img = cv2.erode(thresh_img,kernel,iterations = self.class_parameters['dust_iter'])
        contours, _ = cv2.findContours(thresh_img, 1, 2)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x))
        smallest = [contour for contour in contours if cv2.contourArea(contour) < max_dust_size]
        dust_mask = np.zeros_like(imgray)
        dust_mask = cv2.drawContours(dust_mask, smallest, -1, 255, cv2.FILLED)
        dust_mask = cv2.dilate(dust_mask, kernel, iterations=1)
        return dust_mask
    
    def fill_dust(self, img, dust_mask):
        # work in progress, uses dust threshold image to erase dust
        if len(img.shape) == 3:
            channels = cv2.split(img)
            filled = []
            for channel in channels:
                filled.append(cv2.inpaint(channel, dust_mask, 3, cv2.INPAINT_TELEA))
            filled = cv2.merge(filled)
        else:
            filled = cv2.inpaint(img, dust_mask, 3, cv2.INPAINT_TELEA)
        return filled

    def find_optimal_crop(self):
        # Determines the optimal crop around an image and corrects for misalignment/rotation
        thresh = self.get_threshold(self.RAW_IMG)
        #thresh = self.get_edges(img) # experimental threholding using edge detection
        contours, _ = cv2.findContours(thresh, 1, 2)
        if len(contours) == 0:
            return thresh, None, None # if no contours are found, skip cropping
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour) # bounding box of largest contour
        y, x = self.RAW_IMG.shape[0], self.RAW_IMG.shape[1]
        if rect[2] <= 0:
            rect = ((rect[0][0], rect[0][1]), (rect[1][1], rect[1][0]), rect[2] + 90) # correction for if the rectangle rotation is exactly zero
        rect = ((rect[0][0]/y, rect[0][1]/x), (rect[1][0]/y, rect[1][1]/x), rect[2]) # normalizes crop for different sized images
        return thresh, rect, largest_contour
    
    def crop(self, img, rect, include_EQ_ignore=False):
        if rect is not None:
            if img.shape[0] > img.shape[1]:
                x_crop = self.border_crop
                y_crop = self.border_crop * img.shape[1] / img.shape[0]
            else:
                y_crop = self.border_crop
                x_crop = self.border_crop * img.shape[0] / img.shape[1]
            y, x = img.shape[0], img.shape[1]
            rect = ((rect[0][0]*y, rect[0][1]*x), (rect[1][0]*y, rect[1][1]*x), rect[2]) # convert normalized crop to corresponding pixel coordinates
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            if not (include_EQ_ignore and self.class_parameters['ignore_neg_border']):
                box = self.shrink_box(box, x_crop, y_crop)
            if include_EQ_ignore:
                ignore_border = np.array(self.class_parameters['ignore_border'])
                box = self.shrink_box(box, ignore_border[0], ignore_border[1])
            width = int(rect[1][1] * (1 - x_crop / 100))
            height = int(rect[1][0] * (1 - y_crop / 100))
            src_pts = box.astype('float32')
            dst_pts = np.array([[0, width-1],
                            [0, 0],
                            [height-1, 0],
                            [height-1, width-1]], dtype='float32')
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            img = cv2.warpPerspective(img, M, (height, width))
            if rect[2] > 45:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # rotates image back if it has been rotated
        return img
    
    def get_edges(self, img):
        # experimental, attempt at smarter image thresholding, not used
        img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to b&w
        blurred = cv2.GaussianBlur(imgray, (5,5), 0)
        max_val = 0
        min_val = abs(self.threshold) / 100 * 255
        edges = cv2.Canny(blurred, min_val, max_val)
        kernel = np.ones((5,5),np.uint8)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 50, 10)
        output = np.zeros_like(edges)
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(output, (l[0], l[1]), (l[2], l[3]), 255, 2, cv2.LINE_AA)
        
        output = cv2.dilate(output,kernel,iterations = 8)
        output = cv2.erode(output,kernel,iterations = 8)
        if self.threshold > 0:
            output = 255 - output
        return output

    def get_threshold(self, img):
        # Generates the threshold image used to find contours
        img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to b&w
        dark_threshold = int(self.dark_threshold / 100 * 255)
        _, dark_thresh_img = cv2.threshold(imgray, dark_threshold, 255, 0)
        light_threshold = int(self.light_threshold / 100 * 255)
        _, light_thresh_img = cv2.threshold(imgray, light_threshold, 255, cv2.THRESH_BINARY_INV)
        thresh_img = cv2.bitwise_and(dark_thresh_img, light_thresh_img)
        kernel = np.ones((7,7),np.uint8)
        thresh_img = cv2.erode(thresh_img, kernel, iterations = 2)
        return thresh_img
    
    def rotate(self, img, undo=False):
        # Applies flip and rotation to image
        rotation = self.rotation % 4
        if undo: # option for reverse operation
            if self.flip:
                img = np.ascontiguousarray(img[:,::-1])
            match rotation:
                case 0:
                    pass
                case 1:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                case 2:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                case 3:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            match rotation:
                case 0: 
                    pass
                case 1: 
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                case 2: 
                    img = cv2.rotate(img, cv2.ROTATE_180)
                case 3: 
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if self.flip:
                img = np.ascontiguousarray(img[:,::-1])
        return img
    
    def draw_histogram(self, img):
        # Generates histogram plot of image
        hist_plot = np.zeros(self.class_parameters['histogram_plt_size'], np.uint8) # Base image for the histogram
        width = self.class_parameters['histogram_plt_size'][1]
        height = self.class_parameters['histogram_plt_size'][0] - 10 # Leave a little border from the top
        histograms = []
        maxes = []
        if len(img.shape) == 2: # Determines if the image is grayscale
            channels = [img]
            colours = [(255, 255, 255)]
        else:
            channels = cv2.split(img)
            colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        bins = [256] # The number of divisions in the histogram
        smoothing = 5
        for channel in channels:
            hist = cv2.calcHist([channel], [0], None, bins, [0, 65536]) # Generates the histogram
            maxes.append(np.max(hist)) # Keeps track of the maximum value of the histogram
            hist[1:-1] = cv2.GaussianBlur(hist[1:-1], (smoothing,smoothing), 0) # Smooths out the histogram
            hist = np.squeeze(hist)
            histograms.append(hist)
        
        for (hist, colour) in zip(histograms, colours):
            if np.max(maxes) != 0:
                hist = hist / np.max(maxes) * height # Scales all histograms to fit in the image
            pts = np.stack((np.linspace(0, width, len(hist)), hist), -1).reshape(-1,1,2).squeeze().tolist() # Reformats hist into a list of 2d points
            pts.insert(0, [0, 0])
            pts.append([width, 0])
            new_plot = np.zeros(self.class_parameters['histogram_plt_size'], np.uint8)
            new_plot = cv2.fillPoly(new_plot, np.array([pts]).astype(np.int32), color=colour) # Generates histogram as a polygon
            hist_plot = hist_plot + new_plot # Add current histogram channel to the overall histogram plot
        return hist_plot
    
    def add_frame(self, img):
        if self.class_parameters['frame'] == 0:
            if self.class_parameters['fit_aspect_ratio'] == 'Keep Original':
                return img # skip if aspect ratio and frame have default values

        frame_size = max(1, int(min(img.shape[:2]) * self.class_parameters['frame'] / 100)) # at least one pixel wide frame
        new_shape = (img.shape[0] + 2 * frame_size, img.shape[1] + 2 * frame_size) + img.shape[2:]
        frame_img = np.ones(new_shape, dtype=img.dtype) * 65535
        frame_img[frame_size:-frame_size, frame_size:-frame_size] = img # center image inside frame

        if self.class_parameters['fit_aspect_ratio'] != 'Keep Original': # fit image to aspect ratio
            target_w, target_h = map(int, self.class_parameters['fit_aspect_ratio'].split(' ', 1)[0].split(':')) # parse 'W:H (text)' to W, H
            target_ratio = target_w / target_h
            current_ratio = frame_img.shape[1] / frame_img.shape[0]
            if current_ratio > target_ratio: # image is wider than target aspect ratio
                new_height = int(frame_img.shape[1] / target_ratio)
                pad = (new_height - frame_img.shape[0]) // 2
                padding = ((pad, pad + (new_height - frame_img.shape[0]) % 2), (0, 0)) + (() if img.ndim == 2 else ((0, 0),))
            else:
                new_width = int(frame_img.shape[0] * target_ratio)
                pad = (new_width - frame_img.shape[1]) // 2
                padding = ((0, 0), (pad, pad + (new_width - frame_img.shape[1]) % 2)) + (() if img.ndim == 2 else ((0, 0),))
            frame_img = np.pad(frame_img, padding, mode='constant', constant_values=65535)

        return frame_img

    def clear_memory(self):
        # Deletes instances of images to save memory
        to_del = ['IMG', 'thresh', 'RAW_IMG', 'RAW_LIN', 'ANALYSIS_SRC', 'proxy_RAW_IMG', 'proxy_RAW_LIN', 'dust_mask']
        for attr in to_del:
            if hasattr(self, attr):
                delattr(self, attr)
        self._grid = self._norm = None
        self._norm_key = self._grid_key = None
        self.processed = False

    def __sizeof__(self):
        # Returns the size of the largest numpy arrays
        total = 0
        for attr in self.__dict__.keys():
            if type(getattr(self, attr)) is np.ndarray:
                total += getattr(self, attr).nbytes
        return total

    def _picker_mask(self, x, y, shape):
        # circular mask in the (rotated) preview orientation, returned in the stored orientation
        mask = self.rotate(np.zeros(shape, dtype=np.uint8))
        x = int(x * mask.shape[1])
        y = int(y * mask.shape[0])
        radius = max(1, int(min(mask.shape) * self.class_parameters['picker_radius'] / 100))
        mask = cv2.circle(mask, (x, y), radius, 255, -1)
        return self.rotate(mask, True)

    def set_wb_from_picker(self, x, y):
        # x, y normalized between 0 and 1 as a proportion along the image height and width
        # Neutralizes the picked patch by setting the magenta/yellow filtration from the cached normalized density
        self.process() # makes sure the normalized proxy is current
        if self.film_type in (FILM_TYPE_CROP, FILM_TYPE_BW) or self._norm is None:
            return
        norm = self._norm
        mask = self._picker_mask(x, y, norm.shape[:2]) > 0
        if not mask.any():
            return
        sampled = norm[mask].mean(axis=0)
        analysis = self._analysis
        params = self._params()
        m, yel = calculate_wb_shifts_from_log(sampled, analysis.trimmed_bounds(params))
        self.wb_cyan = 0.0
        self.wb_magenta = float(np.clip(m, -1.0, 1.0))
        self.wb_yellow = float(np.clip(yel, -1.0, 1.0))
        self.process()

    def get_base_colour(self, x, y):
        # x, y normalized between 0 and 1 as a proportion along the image height and width
        # Mean linear (as-shot WB) colour under the picker circle becomes the film base
        mask = self._picker_mask(x, y, self.RAW_LIN.shape[:2]) > 0
        if not mask.any():
            return
        lin = self.RAW_LIN[mask].astype(np.float64) / 65535.0
        gain = self._camera_gain()
        if gain is not None:
            g = np.asarray(gain, dtype=np.float64)[:3]
            lin = lin * (g / g.max())
        self.base_rgb = tuple(float(v) for v in lin.mean(axis=0))

    def blank_base_colour(self):
        # Linear (as-shot WB) film base from a blank scan: mean of the pixels around the 90th percentile brightness
        lin = self.ANALYSIS_SRC.reshape(-1, 3).astype(np.float64) / 65535.0
        gain = self._camera_gain()
        if gain is not None:
            g = np.asarray(gain, dtype=np.float64)[:3]
            lin = lin * (g / g.max())
        brightness = lin.sum(axis=1)
        lo, hi = np.percentile(brightness, (89, 91))
        band = lin[(brightness >= lo) & (brightness <= hi)]
        return tuple(float(v) for v in band.mean(axis=0))

    @staticmethod
    def shrink_box(box, x, y):
        # given box with 4 corner coordinates, returns new box coordinates shrunken by x and y percentages
        sorted = np.sort(box, 0)
        topleft = min(box, key=sum)
        index = np.where(box==topleft)[0][0]
        ordered = np.roll(box, -index, axis=0)
        h = (sorted[2,1] + sorted[3,1] - sorted[0,1] - sorted[1,1]) / 2
        w = (sorted[2,0] + sorted[3,0] - sorted[0,0] - sorted[1,0]) / 2
        skew = (ordered[3,0] - ordered[0,0]) / w * 1.5 # change in y over x-axis
        centre = np.mean(ordered, 0)
        y_offset = y / 100 * h
        x_offset = x / 100 * w
        offset = np.zeros_like(ordered)
        offset[ordered[:,1] < centre[1], 1] += int(y_offset)
        offset[ordered[:,1] > centre[1], 1] -= int(y_offset)
        offset[ordered[:,0] < centre[0], 0] += int(x_offset)
        offset[ordered[:,0] > centre[0], 0] -= int(x_offset)
        for point in offset:
            if point[0] > 0:
                point[1] -= int(x_offset * skew)
            else:
                point[1] += int(x_offset * skew)
            if point[1] < 0:
                point[0] -= int(y_offset * skew)
            else:
                point[0] += int(y_offset * skew)
        new_box = ordered + offset
        new_box = np.roll(new_box, index, axis=0)
        return new_box.astype(np.int32)
