import os
from tkinter import ttk,  filedialog, messagebox, colorchooser
import tkinter as tk
from PIL import ImageTk
import threading
import numpy as np 
import cv2
import multiprocessing
from typing import Literal
from abc import ABC, abstractmethod

#custom classes
from CustomWidgets import ScrollFrame, ScaleEntry, CheckLabel, ComboLabel, MultiEntryLabel
from RawProcessing import RawProcessing
from conversion import crosstalk as xt_profiles
from conversion.modes import ProcessMode
from conversion.papers import profiles_for_mode
from conversion.params import SPEC_DEFAULTS, mode_defaults
from conversion.print_curve import kelvin_to_wb, wb_to_kelvin
from ResourcePaths import resource_path
from AppPaths import ensure_app_data_directory

#logging
import logging

logger = logging.getLogger(__name__)

CONFIG_VERSION = 2

class BaseGUI(ABC):
    def __init__(self, master):
        # Initialize Variables
        self.config_path = ensure_app_data_directory()
        self.photos = []
        self.in_progress = set() # keeps track photos that are in processing when first loading
        self.photo_process_values = ['RAW', 'Threshold', 'Contours', 'Histogram', 'Full Preview']
        self.filetypes = ['TIFF', 'PNG', 'JPG'] # Export File Types
        self.fit_aspect_ratios = ['Keep Original', '16:9 (Landscape)', '3:2 (Landscape)', '4:3 (Landscape)', '5:4 (Landscape)', '1:1 (Square)', '4:5 (Portrait)', '3:4 (Portrait)', '2:3 (Portrait)', '9:16 (Portrait)']
        self.allowable_image_filetypes = [
            ('RAW files', '*.DNG *.CR2 *.CR3 *.NEF *.ARW *.RAF *.ORF *.ORI *.ERF *.GPR *.RAW *.CRW *.RW2 *.dng *.cr2 *.cr3 *.nef *.arw *.raf *.orf *.ori *.erf *.grp *.raw *.crw *.rw2'),
            ('Image files', '*.PNG *.JPG *.JPEG *.BMP *.TIFF *.TIF *.png *.jpg *.jpeg *.bmp *.tiff *.tif')
            ]
        self.header_style = ('Segoe UI', 10, 'normal') # Defines font for section headers
        self.wb_picker = False
        self.base_picker = False
        self.unsaved = False # Indicates that settings have changed need to be saved

        self.default_advanced_settings = dict(
            max_processors_override = 0, # 0 means disabled
            preload = 4 # Buffer size to preload photos
        )
        self.advanced_settings = self.default_advanced_settings.copy()

        self.default_settings = dict(
            film_type = 0,
            dark_threshold = 25,
            light_threshold = 100,
            border_crop = 1,
            flip = False,
            base_detect = 0,
            base_rgb = None, # linear film base picked from the scan, None = Auto
            remove_dust = False,
            **SPEC_DEFAULTS
        )

        self.global_settings = self.default_settings.copy()
        self.master = master
        # Building the GUI
        self.master.title('Film Scan Converter')
        try:
            self.master.state('zoomed')
        except Exception as e: # Exception for linux
            m = self.master.maxsize()
            self.master.geometry('{}x{}+0+0'.format(*m))
            logger.exception(f'Exception: {e}')
        self.master.geometry('800x500')

        menubar = tk.Menu(self.master, relief=tk.FLAT)
        self.filemenu = tk.Menu(menubar, tearoff=0)
        self._build_file_menu_controls()
        self.filemenu.add_command(label='Save Settings', command=self.save_settings, accelerator='Ctrl+S')
        self.filemenu.add_separator()
        self.filemenu.add_command(label='Exit', command=self.on_closing)
        menubar.add_cascade(label='File', menu=self.filemenu)
        self.editmenu = tk.Menu(menubar, tearoff=0)
        self.editmenu.add_command(label='Copy Settings', command=self.copy_settings, accelerator='Ctrl+C')
        self.editmenu.add_command(label='Paste Settings', command=self.paste_settings, state=tk.DISABLED, accelerator='Ctrl+V')
        self.editmenu.add_separator()
        self.editmenu.add_command(label='Reset to Default Settings', command=self.reset_settings)
        self.editmenu.add_separator()
        self.editmenu.add_command(label='Advanced Settings...', command=self.advanced_dialog)
        menubar.add_cascade(label='Edit', menu=self.editmenu)
        self.master.config(menu=menubar)

        mainFrame = ttk.Frame(self.master, padding=10)
        mainFrame.pack(side=tk.TOP, anchor=tk.NW, fill='both', expand=True)
        mainFrame.grid_rowconfigure(0, weight=1)
        mainFrame.grid_columnconfigure(1, weight=1)

        self.controlsFrame = ttk.Frame(mainFrame)
        self.controlsFrame.grid(row=0, column=0, sticky='NS', rowspan=10)
        self.controlsFrame.grid_rowconfigure(0, weight=1)
        self.controlsFrame.grid_columnconfigure(0, weight=1)
        dynamic_scroll_frame = ScrollFrame(self.controlsFrame)

        self.widgets = {} # stores dictionary of all widgets

        # Importing RAW scans
        import_title = ttk.Label(text='Select Photo', font=self.header_style, padding=2)
        importFrame = ttk.LabelFrame(dynamic_scroll_frame.frame, borderwidth=2, labelwidget=import_title, padding=5)
        importFrame.grid(row=0, column=0, sticky='EW')
        self._build_import_controls(importFrame)

        # Processing Frame
        processing_title = ttk.Label(text='Processing Settings', font=self.header_style, padding=2)
        processingFrame = ttk.LabelFrame(dynamic_scroll_frame.frame, borderwidth=2, labelwidget=processing_title, padding=5)
        processingFrame.grid(row=1, column=0, sticky='EW')
        # Reject Checkbutton to exclude from export
        rejectFrame = ttk.Frame(processingFrame)
        rejectFrame.grid(row=0, column=0, sticky='EW')
        self.reject_check = CheckLabel(rejectFrame, 'Reject Photo:', 0, 'reject', self.widgets, False, command=lambda widget: self.widget_changed(widget, 'update'))
        # Selection of global settings
        self.globFrame = ttk.Frame(processingFrame)
        self.globFrame.grid(row=1, column=0, sticky='EW')
        self.glob_check = CheckLabel(self.globFrame, 'Sync with Global Settings:', 0, default_value=True, global_sync=False, command=self.set_global)
        # Selection of film type
        self.filmFrame = ttk.Frame(processingFrame)
        self.filmFrame.grid(row=3, column=0, sticky='EW')
        film_types = ['Black & White Negative', 'Colour Negative', 'Slide (Colour Positive)','Crop Only (RAW)']
        self.film_type = ComboLabel(self.filmFrame, 'Film Type:', 0, film_types, 'film_type', self.widgets, width=25, command=self.film_type_changed)
        # Dust removal
        self.dustFrame = ttk.Frame(processingFrame)
        self.dustFrame.grid(row=4, column=0, sticky='EW')
        CheckLabel(self.dustFrame, 'Remove Dust:', 0, 'remove_dust', self.widgets, command=lambda widget: self.widget_changed(widget, 'update'))

        # Automatic Cropping Settings
        controls_title = ttk.Label(text='Automatic Crop & Rotate', font=self.header_style, padding=2)
        self.cropFrame = ttk.LabelFrame(dynamic_scroll_frame.frame, borderwidth=2, labelwidget=controls_title, padding=5)
        self.cropFrame.grid(row=4, column=0, sticky='EW')
        crop_adjustments = ttk.Frame(self.cropFrame)
        crop_adjustments.pack(fill='x')
        crop_adjustments.grid_rowconfigure(1, weight=1)
        crop_adjustments.grid_columnconfigure(3, weight=1)
        ScaleEntry(crop_adjustments, 'Dark Threshold:', 0, 0, 100, 'dark_threshold', self.widgets, command=self.widget_changed)
        ScaleEntry(crop_adjustments, 'Light Threshold:', 1, 0, 100, 'light_threshold', self.widgets, command=self.widget_changed)
        ScaleEntry(crop_adjustments, 'Border Crop (%):', 2, -5, 20, 'border_crop', self.widgets, command=self.widget_changed)
        self.flip_check = CheckLabel(crop_adjustments, 'Flip Horizontally:', 3, 'flip', self.widgets, command=self.set_flip)
        rotButtons = ttk.Frame(self.cropFrame)
        rotButtons.pack(fill='x')
        ttk.Button(rotButtons, text='Rotate Counterclockwise', width=22, command=self.rot_counterclockwise).pack(side=tk.LEFT, padx=2, pady=5)
        self.set_tooltip(rotButtons.winfo_children()[-1], "(Shift+R)")
        ttk.Button(rotButtons, text='Rotate Clockwise', width=22, command=self.rot_clockwise).pack(side=tk.LEFT, padx=2, pady=5)
        self.set_tooltip(rotButtons.winfo_children()[-1], "(R)")

        # Colour settings
        picker = tk.PhotoImage(
            file=resource_path(os.path.join('assets', 'dropper.png'))
        ).subsample(15,15)
        colour_title = ttk.Label(text='Colour Adjustment', font=self.header_style, padding=2)
        self.colourFrame = ttk.LabelFrame(dynamic_scroll_frame.frame, borderwidth=2, labelwidget=colour_title, padding=5)
        colour_controls = ttk.Frame(self.colourFrame)
        colour_controls.pack(fill='x', side=tk.LEFT)
        base_setting_modes = ['Auto Detect','Set Manually']
        self.base_mode = ComboLabel(colour_controls, 'Film Base Colour', 0, base_setting_modes, 'base_detect', self.widgets, command=self.set_base_detect)
        self.base_clr_lbl = ttk.Label(colour_controls, text='RGB:')
        self.base_rgb = tk.StringVar()
        self.base_rgb_lbl = ttk.Label(colour_controls, textvariable=self.base_rgb)
        self.rgb_display = tk.Frame(colour_controls, bg=self._from_rgb((128, 128, 128)), height=20, width=20, relief=tk.GROOVE, borderwidth=1)
        self.base_pick_button = ttk.Button(colour_controls, image=picker, command=self.set_base)
        self.base_pick_button.image = picker
        self.base_buttons_frame = ttk.Frame(colour_controls)
        ttk.Button(self.base_buttons_frame, text='Set RGB', command=lambda:self.set_base_rgb(0), width=8).pack(side=tk.LEFT, padx=2, pady=2, anchor='center')
        ttk.Button(self.base_buttons_frame, text='Import Blank...', command=lambda:self.set_base_rgb(2), width=13).pack(side=tk.LEFT, padx=2, pady=2, anchor='center')
        self._set_base_display(self.global_settings.get('base_rgb'))
        ttk.Label(colour_controls, text='White Balance Picker:').grid(row=3, column=0, sticky=tk.E)
        self.wb_picker_button = ttk.Button(colour_controls, image=picker, command=self.pick_wb)
        self.wb_picker_button.grid(row=3, column=1, sticky=tk.W)
        self.wb_picker_button.image = picker
        skip = lambda widget: self.widget_changed(widget, 'skip crop')
        self.temperature = ScaleEntry(colour_controls, 'Temperature (K):', 4, 3000, 12000, None, None, global_sync=False, increment=100, default_value=5500, command=self.temperature_changed)
        ScaleEntry(colour_controls, 'Cyan:', 5, -1, 1, 'wb_cyan', self.widgets, is_float=True, increment=0.01, command=skip)
        ScaleEntry(colour_controls, 'Magenta:', 6, -1, 1, 'wb_magenta', self.widgets, is_float=True, increment=0.01, command=self.filtration_changed)
        ScaleEntry(colour_controls, 'Yellow:', 7, -1, 1, 'wb_yellow', self.widgets, is_float=True, increment=0.01, command=self.filtration_changed)
        ScaleEntry(colour_controls, 'Saturation:', 8, 0.5, 1.5, 'dye_separation', self.widgets, is_float=True, increment=0.05, command=skip)
        ScaleEntry(colour_controls, 'Cast Removal:', 9, 0, 1, 'cast_removal', self.widgets, is_float=True, increment=0.05, command=skip)
        stocks = xt_profiles.list_profiles()
        ComboLabel(colour_controls, 'Film Stock:', 10, stocks, 'crosstalk_profile', self.widgets, output_list=stocks, width=25, command=skip)
        ScaleEntry(colour_controls, 'Crosstalk Strength:', 11, 0, 1, 'crosstalk_strength', self.widgets, is_float=True, increment=0.05, command=skip)

        # Brightness Settings
        brightness_title = ttk.Label(text='Print Adjustment', font=self.header_style, padding=2)
        self.exposureFrame = ttk.LabelFrame(dynamic_scroll_frame.frame, borderwidth=2, labelwidget=brightness_title, padding=5)
        self.exposureFrame.grid(row=7, column=0, sticky='EW')
        exposureControls = ttk.Frame(self.exposureFrame)
        exposureControls.pack(fill='x', side=tk.LEFT)
        exposureControls.grid_columnconfigure(3, weight=1)
        self._print_row = 0
        def row():
            self._print_row += 1
            return self._print_row - 1
        def fslider(label, key, lo, hi, step):
            return ScaleEntry(exposureControls, label, row(), lo, hi, key, self.widgets, is_float=isinstance(step, float), increment=step, command=skip)
        fslider('Print Density:', 'density', 0.0, 2.0, 0.05)
        fslider('Grade (ISO R):', 'grade', 50, 180, 1)
        self.effective_grade_lbl = ttk.Label(exposureControls, text='')
        self.effective_grade_lbl.grid(row=row(), column=0, columnspan=3, sticky='e')
        CheckLabel(exposureControls, 'Auto Density:', row(), 'auto_density', self.widgets, command=skip)
        CheckLabel(exposureControls, 'Auto Grade:', row(), 'auto_grade', self.widgets, command=skip)
        fslider('Toe:', 'toe', -1.0, 1.0, 0.05)
        fslider('Toe Width:', 'toe_width', 0.1, 5.0, 0.1)
        fslider('Shoulder:', 'shoulder', -1.0, 1.0, 0.05)
        fslider('Shoulder Width:', 'shoulder_width', 0.1, 5.0, 0.1)
        fslider('Shadows:', 'shadow_density', -0.9, 0.9, 0.01)
        fslider('Highlights:', 'highlight_density', -0.5, 0.5, 0.01)
        self.paper_keys = ['neutral']
        # UI-only toggle (no key: not a conversion parameter), paper controls hidden by default
        self.paper_toggle = CheckLabel(exposureControls, 'Paper Settings:', row(), command=self.toggle_paper_settings)
        self.paper_combo = ComboLabel(exposureControls, 'Paper:', row(), ['Neutral'], 'paper_profile', self.widgets, output_list=self.paper_keys, width=25, command=skip)
        CheckLabel(exposureControls, 'Paper White:', row(), 'paper_dmin', self.widgets, command=skip)
        CheckLabel(exposureControls, 'Paper Black:', row(), 'paper_black', self.widgets, command=skip)
        self.paper_widgets = [self.paper_combo, self.widgets['paper_dmin'], self.widgets['paper_black']]
        for widget in self.paper_widgets:
            widget.hide()
        fslider('White Point:', 'white_point_offset', -0.25, 0.25, 0.01)
        fslider('Black Point:', 'black_point_offset', -0.25, 0.25, 0.01)
        self.clip_widgets = [
            fslider('Luma Clip:', 'luma_clip_pos', -100, 100, 1),
            fslider('Colour Clip:', 'color_clip_pos', -100, 100, 1),
        ]

        # Export photo settings
        export_title = ttk.Label(text='Export Settings', font=self.header_style, padding=2)
        export_frame = ttk.LabelFrame(dynamic_scroll_frame.frame, borderwidth=2, labelwidget=export_title, padding=5)
        export_frame.grid(row=9, column=0, sticky='EW')
        export_settings_frame = ttk.Frame(export_frame)
        export_settings_frame.pack(fill='x')
        self.export_filetype_widget = ComboLabel(export_settings_frame, 'Export File Type:', 0, self.filetypes, 'filetype', global_sync=False, output_list=self.filetypes, command=lambda widget:self.widget_changed(widget, 'skip', False), default_value=RawProcessing.default_parameters['filetype'])
        self.frame = ScaleEntry(export_settings_frame, 'White Frame (%):', 1, 0, 10, 'frame', global_sync=False, command=lambda widget:self.widget_changed(widget, 'update', False), default_value=RawProcessing.default_parameters['frame'])
        ComboLabel(export_settings_frame, 'Fit Aspect Ratio:', 2, self.fit_aspect_ratios, 'fit_aspect_ratio', global_sync=False, output_list=self.fit_aspect_ratios, command=lambda widget:self.widget_changed(widget, 'update', False), default_value=RawProcessing.default_parameters['fit_aspect_ratio'], width=20)
        export_settings_frame.pack(fill='x', pady=(0, 15)) # adds some spacing
        self._build_export_controls(export_frame)

        # Progress Bar
        self.progressFrame = ttk.Frame(self.controlsFrame, padding=2)
        self.progress_percentage = ttk.Label(self.progressFrame)
        self.progress_percentage.pack(side=tk.RIGHT, anchor='n')
        self.progress = tk.DoubleVar()
        self.progress.set(0)
        self.progress_bar = ttk.Progressbar(self.progressFrame, variable=self.progress)
        self.progress_bar.pack(fill='x')
        self.progress_msg = ttk.Label(self.progressFrame)
        self.progress_msg.pack(side=tk.LEFT)

        # Showing converted image preview and intermediary steps
        self.outputFrame = ttk.Frame(mainFrame)
        self.outputFrame.grid_rowconfigure(3, weight=1)
        self.outputFrame.grid_columnconfigure(0, weight=1)
        self.read_error_lbl = ttk.Label(mainFrame, text='Error: File could not be read', font=('Segoe UI', 20), justify='center', anchor='center')
        self.read_error_lbl.bind('<Configure>', lambda e: self.read_error_lbl.config(wraplength=self.read_error_lbl.winfo_width()))

        # Process showing
        process_select_Frame = ttk.Frame(self.outputFrame)
        process_select_Frame.grid(row=0, column=0, pady=3)
        process_select_Frame.grid_rowconfigure(0, weight=1)
        process_select_Frame.grid_columnconfigure(1, weight=1)
        ttk.Label(process_select_Frame, text='Show:').grid(row=0, column=0)
        self.photo_process_Combo = ttk.Combobox(process_select_Frame, state='readonly', values=self.photo_process_values)
        self.photo_process_Combo.current(0)
        self.photo_process_Combo.bind('<<ComboboxSelected>>', self.update_IMG)
        self.photo_process_Combo.grid(row=0, column=1, padx=2)
        self.set_tooltip(self.photo_process_Combo, "(Q) RAW\n(W) Threshold\n(E) Contours\n(H) Histogram\n(F) Full Preview")
        self.process_photo_frame = tk.Frame(self.outputFrame, padx=3, pady=3)
        self.process_photo_frame.grid(row=1, column=0)
        self.process_photo = ttk.Label(self.process_photo_frame)
        self.process_photo.pack()

        # Converted Preview
        ttk.Label(self.outputFrame, text='Preview:', font=self.header_style).grid(row=2, column=0)
        self.result_photo_frame = tk.Frame(self.outputFrame, padx=3, pady=3)
        self.result_photo_frame.grid(row=3, column=0)
        self.result_photo = ttk.Label(self.result_photo_frame)
        self.result_photo.pack()

        # Bindings
        self.master.bind('<Configure>', self.resize_event)
        self.master.bind('<Key>', self.key_handler)
        self.master.bind('<Button>', self.click)
        self.master.bind('<Control-c>', self.copy_settings)
        self.master.bind('<Control-v>', self.paste_settings)
        self.master.bind('<Control-s>', self.save_settings)
        self.master.protocol('WM_DELETE_WINDOW', self.on_closing)

        dynamic_scroll_frame.update()
        self._update_photo_navigation()

        # Loading advanced parameters from the config file
        try:
            params_dict = np.load(os.path.join(self.config_path,'config.npy'), allow_pickle=True).item()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.exception(f'Exception: {e}')
        else:
            if params_dict.get('config_version', 1) < CONFIG_VERSION:
                params_dict.pop('ignore_border', None) # v1 meant "EQ ignore border"; the analysis border has a new default
            for settings in [self.advanced_settings, RawProcessing.class_parameters]:
                for attr in settings:
                    if attr in params_dict:
                        settings[attr] = params_dict[attr] # Initializes every parameter with imported parameters
                    else:
                        logger.info(f'Attribute {attr} not found in imported config.npy file.')
                    
        for widget in self.widgets.values():
            if widget.key in self.default_settings:
                widget.set(self.default_settings[widget.key]) # initializes widgets with default settings

    @abstractmethod
    def _build_file_menu_controls(self):
        pass

    @abstractmethod
    def _build_import_controls(self, import_frame):
        pass

    @abstractmethod
    def _build_export_controls(self, export_frame):
        pass

    @abstractmethod
    def _current_photo_index(self):
        pass

    @abstractmethod
    def _set_import_controls_enabled(self, enabled):
        pass

    @abstractmethod
    def _set_imported_photo_names(self, photo_names):
        pass

    @abstractmethod
    def _update_photo_navigation(self):
        pass

    @abstractmethod
    def _update_export_controls(self):
        pass

    @abstractmethod
    def _individual_export_filename(self):
        pass

    @abstractmethod
    def _set_individual_export_controls_enabled(self, enabled):
        pass

    def _individual_export_progress_message(self):
        return 'Exporting photo...'

    def _after_individual_export(self):
        pass

    def _load_photo(self, photo, full_res=False):
        photo.load(full_res)

    def _export_photo(self, photo, filename):
        photo.export(filename)

    def advanced_dialog(self):
        # Pop-up window to contain all the advanced settings that don't fit in the main controls
        def apply_settings():
            # applies the values stored in the widgets to the rest of the class
            for widget in advanced_widgets.values():
                if widget.key in RawProcessing.class_parameters:
                    RawProcessing.class_parameters[widget.key] = widget.get()
                else:
                    self.advanced_settings[widget.key] = widget.get()
            
            quit()
            for photo in self.photos:
                photo.clear_memory()
            self.load_IMG()
        
        def save_settings():
            # saves the advanced settings as a config.npy file
            apply_settings()
            params_dict = dict()
            for attr in RawProcessing.advanced_attrs:
                params_dict[attr] = RawProcessing.class_parameters[attr]
            for attr in self.advanced_settings:
                params_dict[attr] = self.advanced_settings[attr]
            params_dict['config_version'] = CONFIG_VERSION
            np.save(f'{os.path.join(self.config_path,"config.npy")}', params_dict)
        
        def quit():
            self.master.attributes('-topmost', 0)
            top.destroy()

        def set_widgets(settings_dict):
            # Applies the settings stored in the settings_dict to the advanced GUI
            for widget in advanced_widgets.values():
                if widget.key in settings_dict:
                    widget.set(settings_dict[widget.key])
        
        def reset():
            # Resets advanced settings to default values
            set_widgets(RawProcessing.default_parameters)
            set_widgets(self.default_advanced_settings)

        top = tk.Toplevel(self.master)
        top.transient(self.master)
        top.title('Advanced Settings')
        top.grab_set()
        top.bind('<Button>', lambda event: event.widget.focus_set())
        top.resizable(False, False)
        top.focus_set()
        top.protocol('WM_DELETE_WINDOW', quit)

        mainFrame = ttk.Frame(top, padding=10)
        mainFrame.pack(fill='x')
        firstColumn = ttk.Frame(mainFrame, padding=2)
        firstColumn.grid(row=0, column=0, sticky='n')
        import_lbl = ttk.Label(top, text='Import:', font=self.header_style, padding=2)
        import_settings = ttk.LabelFrame(firstColumn, borderwidth=2, labelwidget=import_lbl, padding=5)
        import_settings.pack(fill='x', expand=True)
        export_lbl = ttk.Label(top, text='Export:', font=self.header_style, padding=2)
        export_settings = ttk.LabelFrame(firstColumn, borderwidth=2, labelwidget=export_lbl, padding=5)
        export_settings.pack(fill='x', expand=True)
        secondColumn = ttk.Frame(mainFrame, padding=2)
        secondColumn.grid(row=0, column=1, sticky='n')
        process_lbl = ttk.Label(top, text='Processing:', font=self.header_style, padding=2)
        process_settings = ttk.LabelFrame(secondColumn, borderwidth=2, labelwidget=process_lbl, padding=5)
        process_settings.pack(fill='x', expand=True)
        dust_lbl = ttk.Label(top, text='Dust Removal:', font=self.header_style, padding=2)
        dust_settings = ttk.LabelFrame(secondColumn, borderwidth=2, labelwidget=dust_lbl, padding=5)
        dust_settings.pack(fill='x', expand=True)

        advanced_widgets = {}

        # Building pop-up GUI
        allowable_dm_algs = (0, 1, 2, 3, 4, 11, 12)
        dm_algs_list = ('LINEAR','VNG','PPG','AHD','DCB','DHT','AAHD')
        ComboLabel(import_settings, 'Demosaicing Algorithm:', 0, dm_algs_list, 'dm_alg', advanced_widgets, output_list=allowable_dm_algs, width=25)
        MultiEntryLabel(import_settings, 'Median Filter Passes:', 1, 0, 10, 1, key='median_filter_passes', widget_dictionary=advanced_widgets, width=25)
        ComboLabel(import_settings, 'Input Encoding (16-bit, no profile):', 2, ('Auto','sRGB','Linear'), 'input_encoding', advanced_widgets, output_list=('Auto','sRGB','Linear'), width=25)
        fbdd_nr_list = ('Off','Light','Full')
        ComboLabel(import_settings, 'FBDD Noise Reduction:', 5, fbdd_nr_list, 'fbdd_nr', advanced_widgets, width=25)
        MultiEntryLabel(import_settings, 'Noise Threshold', 6, 0, 1000, 1, key='noise_thr', widget_dictionary=advanced_widgets, increment=50, width=25)

        MultiEntryLabel(process_settings, 'Max Proxy Size (W + H):', 0, 500, 20000, 1, key='max_proxy_size', widget_dictionary=advanced_widgets, increment=500)
        MultiEntryLabel(process_settings, 'Photo Preload Buffer Size:', 1, 0, 20, 1, key='preload', widget_dictionary=advanced_widgets)
        MultiEntryLabel(process_settings, 'Analysis Border % (W, H)', 2, 0, 40, 2, key='ignore_border', widget_dictionary=advanced_widgets)
        CheckLabel(process_settings, 'Analysis Includes Negative Border:', 3, 'ignore_neg_border', advanced_widgets, default_value=True)
        MultiEntryLabel(process_settings, 'Colour Picker Radius (%)', 6, 0.5, 10, 1, key='picker_radius', widget_dictionary=advanced_widgets, is_float=True, increment=0.5)

        MultiEntryLabel(dust_settings, 'Threshold:', 0, 0, 50, 1, key='dust_threshold', widget_dictionary=advanced_widgets)
        MultiEntryLabel(dust_settings, 'Noise Closing Iterations:', 1, 1, 10, 1, key='dust_iter', widget_dictionary=advanced_widgets)
        MultiEntryLabel(dust_settings, 'Max Particle Area:', 2, 0, 100, 1, key='max_dust_area', widget_dictionary=advanced_widgets)

        ComboLabel(export_settings, 'Output Colour Space:', 3, ('sRGB','Adobe RGB'), 'output_colour_space', advanced_widgets, False, output_list=('sRGB','Adobe RGB'), width=20)
        MultiEntryLabel(export_settings, 'JPEG Quality:', 0, 0, 100, 1, key='jpg_quality', widget_dictionary=advanced_widgets, increment=10)
        tiff_comp_list = ['No Compression','Lempel-Ziv & Welch','Adobe Deflate (ZIP)','PackBits']
        tiff_comp_out = [1, 5, 8, 32773]
        ComboLabel(export_settings, 'TIFF Compression:', 1, tiff_comp_list, 'tiff_compression', advanced_widgets, False, output_list=tiff_comp_out, width=20)
        MultiEntryLabel(export_settings, 'Max Processors Override:', 2, 0, multiprocessing.cpu_count(), 1, key='max_processors_override', widget_dictionary=advanced_widgets)

        set_widgets(RawProcessing.class_parameters)
        set_widgets(self.advanced_settings)

        buttonFrame = ttk.Frame(mainFrame)
        buttonFrame.grid(row=1, column=0, columnspan=2, sticky='e')
        ttk.Button(buttonFrame, text='Cancel', command=quit).pack(side=tk.RIGHT, padx=2, pady=5, anchor='sw')
        ttk.Button(buttonFrame, text='Save', command=save_settings).pack(side=tk.RIGHT, padx=2, pady=5, anchor='sw')
        ttk.Button(buttonFrame, text='Reset to Default', command=reset, width=17).pack(side=tk.RIGHT, padx=2, pady=5, anchor='sw')

        # Centres pop-up window over self.master window
        top.update_idletasks()
        x = self.master.winfo_x() + int((self.master.winfo_width()/2) - (top.winfo_width()/2))
        y = self.master.winfo_y() + int((self.master.winfo_height()/2) - (top.winfo_height()/2))
        top.geometry('+%d+%d' % (x, y))

    def widget_changed(self, widget, mode: Literal['normal','skip crop','update','skip']='normal', instance=True):
        # called whenever widget is changed
        # applies value stored in widget to photo
        # mode: some setting changes don't need to fully reprocess the photo, this speeds up processing
        # instance: whether the variable is changing an instance variable or class variable
        value = widget.get()
        key = widget.key
        if self.glob_check.get() and widget.global_sync:
            self.global_settings[key] = value
            self.changed_global_settings()
        self.update_UI()
        if instance:
            if hasattr(self, 'current_photo'):
                setattr(self.current_photo, key, value) # change instance
        else:
            RawProcessing.class_parameters[key] = value
        if len(self.photos) == 0:
            return
        match mode:
            case 'normal':
                self.current_photo.process()
                self.update_IMG()
                self.unsaved = True
            case 'skip crop':
                self.current_photo.process(skip_crop=True)
                self.update_IMG()
                self.unsaved = True
            case 'update':
                self.update_IMG()
                self.unsaved = True
            case 'skip':
                pass
    
    def import_from_filenames(self, filenames):
        # Import all files from the given string list
        self.show_progress('Initializing import...')  # display progress opening
        self._set_import_controls_enabled(False)

        total = len(filenames)
        self.photos = []
        photo_names = []

        self.update_progress(20, f'Initializing {str(total)} photos...')
        
        for i, filename in enumerate(filenames):
            print(f'Loading file: {filename}')
            photo = RawProcessing(
                file_directory=filename,
                default_settings=self.default_settings,
                global_settings=self.global_settings,
                config_path=self.config_path
            )
            self.photos.append(photo)
            photo_names.append(f'{str(i + 1)}. {str(photo)}')

        self.update_progress(80, 'Configuring GUI...')
        
        self._set_imported_photo_names(photo_names)

        self.update_progress(90, 'Loading photo...')
        self.load_IMG()  # configure GUI to display the first photo
        self.update_progress(99)
        self._set_import_controls_enabled(True)
        self.update_UI()
        if (
            self.glob_check.get() and self.global_settings != self.default_settings
        ):  # check if global settings are different from default on import
            self.unsaved = (
                True  # if it is the default, then it doesn't need to be saved
            )
        else:
            self.unsaved = False
        self.hide_progress()
    
    def load_IMG(self, event=None):
        # Loading new image into GUI, loads other images in background
        def load_async(photo, i):
            # Threading function to load photo in background
            # photo: RawProcessing object
            # i: the index of the photo in self.photos
            self._load_photo(photo)
            self.in_progress.remove(i)

        if len(self.photos) == 0:
            return

        photo_index = self._current_photo_index()
        self.current_photo = self.photos[photo_index]
        self.glob_check.set(self.current_photo.use_global_settings)
        if self.current_photo.use_global_settings:
            self.apply_settings(self.current_photo, self.global_settings)
        self.change_settings() # Ensures that the GUI matches the photo's parameters
        if not self.current_photo.processed:
            if not hasattr(self.current_photo, 'RAW_IMG'):
                self._load_photo(self.current_photo)
            self.current_photo.process() # only process photos if needed
        self.update_IMG()

        # conservatively load extra images in background to speed up switching, while saving memory
        for i, photo in enumerate(self.photos):
            if (abs(i - photo_index) <= self.advanced_settings['preload']) and not hasattr(photo, 'RAW_IMG') and i not in self.in_progress: # preload photos in buffer ahead or behind of the currently selected one
                threading.Thread(target=load_async, args=(photo, i), daemon=True).start()
                self.in_progress.add(i) # keeps track of photos in progress
            elif (abs(i - photo_index) > self.advanced_settings['preload']) and hasattr(photo, 'RAW_IMG'): # delete photos outside of buffer
                photo.clear_memory()

        self._update_photo_navigation()
    
    def update_IMG(self, full_res=True):
        # Loads new image into GUI
        # Queues full res picture to be loaded when ignore_full_res is set to False and the Full Preview is selected (default behaviour)
        def update_full_res():
            # Checks periodically if the thread is finished, then updates the image
            def check_if_done(t):
                if not t.is_alive():
                    if not self.current_photo.active_processes:
                        self.update_IMG(False) # update image, but do not queue full res process again
                else:
                    self.master.after(300, check_if_done, t) # if not done, wait some time, then check again
            t = threading.Thread(target=self.current_photo.process, args=[True, True, True], daemon=True) # Thread to generate full res previews
            t.start()
            self.master.after_idle(check_if_done, t)
        if len(self.photos) == 0:
            return

        if self.current_photo.get_preview_img() is None:
            self.outputFrame.grid_forget()
            self.read_error_lbl.grid(row=0, column=1, sticky='EW') # displays error message when image cannot be loaded
            return

        self.read_error_lbl.grid_forget()
        self.outputFrame.grid(row=0, column=1, sticky='NSEW')
        if self.photo_process_Combo.current() == 4: # if "Full Preview" is selected
            self.process_photo_frame.grid_forget() # hide the process photo
            self.photo_display_height = max(int(self.master.winfo_height() - 100), 100) # resize image to full display height
        else:
            self.photo_display_height = max(int((self.master.winfo_height() - 100) / 2), 100)
            process_photo = ImageTk.PhotoImage(self.resize_IMG(self.current_photo.get_preview_img(self.photo_process_values[self.photo_process_Combo.current()])))
            self.process_photo.configure(image=[process_photo])
            self.process_photo.image = process_photo
            self.process_photo_frame.grid(row=1, column=0)

        result_photo = ImageTk.PhotoImage(self.resize_IMG(self.current_photo.get_preview_img()))
        self.result_photo.configure(image=[result_photo])
        self.result_photo.image = result_photo
        self._update_effective_grade()
        self.master.update_idletasks()

        if full_res:
            # Generates full resolution image in the background
            if self.photo_process_Combo.current() == 4: # Only display when full preview is selected
                try: 
                    self.master.after_cancel(self.start_full_res)
                except: 
                    pass
                finally: 
                    self.start_full_res = self.master.after(500, update_full_res) # waits for 0.5 s of inactivity before processing

    def resize_IMG(self, img):  
        # Resizes the displayed image based on maximum allowable dimensions, while maintaining aspect ratio
        w, h = img.size
        scale = min(self.photo_display_width / w, self.photo_display_height / h)
        new_img = img.resize((int(w * scale), int(h * scale)))
        return new_img
    
    def resize_event(self, event=None):
        # Attempts to resize the images if the resize event has not been called for 100 ms
        try:
            self.master.after_cancel(self.resize)
        except: 
            pass
        finally:
            self.resize = self.master.after(100, self.resize_UI)
    
    def resize_UI(self):
        # Calculates the maximum dimensions for the displayed images based on the size of the window
        self.photo_display_width = max(int(self.master.winfo_width() - self.controlsFrame.winfo_width() - 20), 100)
        if self.photo_process_Combo.current() == 4:
            self.photo_display_height = max(int(self.master.winfo_height() - 100), 100)
        else:
            self.photo_display_height = max(int((self.master.winfo_height() - 100) / 2), 100)
        self.update_IMG(False)
    
    def previous(self):
        pass

    def next(self):
        pass

    def set_global(self, event=None):
        # Defines behaviour of the 'Sync with Global Settings' checkbox
        if len(self.photos) == 0:
            return
        self.current_photo.use_global_settings = self.glob_check.get()
        self.apply_settings(self.current_photo, self.global_settings)
        self.change_settings()
        self.unsaved = True
        if self.glob_check.get():
            self.current_photo.process()
            self.update_IMG()

    def change_settings(self):
        # Configures GUI to reflect current applied settings for the photo
        for widget in self.widgets.values():
            widget.set(getattr(self.current_photo, widget.key))

        if self.current_photo.FileReadError:
            self.reject_check.disable()
        else:
            self.reject_check.enable()

        self._set_base_display(self.current_photo.base_rgb)
        self._refresh_papers()
        self._sync_temperature()

        self.update_UI()

    def apply_settings(self, photo, settings):
        # applies settings to photo based on input dictionary containing settings
        for attribute in settings.keys():
            setattr(photo, attribute, settings[attribute])
    
    def changed_global_settings(self):
        # sets flags of all photos using global settings to be unprocessed when global settings are changed
        if len(self.photos) == 0:
            return
        for photo in self.photos:
            if photo.use_global_settings:
                photo.processed = False

    def reset_settings(self):
        # Reset settings to default parameters
        if len(self.photos) == 0:
            for widget in self.widgets.values():
                if widget.key in self.default_settings:
                    widget.set(self.default_settings[widget.key])
            return
        if self.glob_check.get():
            affected = sum([photo.use_global_settings for photo in self.photos]) # calculate the total number of photos using global settings
            if affected > 1:
                if not messagebox.askyesno('Reset to Default Settings',f'You are about to globally reset {str(affected)} photos\'s settings.\nDo you wish to continue?', icon='warning'):
                    return
            self.global_settings = self.default_settings.copy() # reset global settings
            self.changed_global_settings()
        self.apply_settings(self.current_photo, self.default_settings) # apply new settings to photo
        self.change_settings() # update GUI with new settings
        self.current_photo.process()
        self.update_IMG()
        self.unsaved = True
    
    # The following functions all define the behaviour of the interactable GUI, such as buttons, entries, and scales
    
    def update_UI(self):
        # Changes which settings are visible
        if (self.film_type.get() == 3) or self.reject_check.get():
            self.exposureFrame.grid_forget() # Hide the exposure frame when set to output RAW
        else:
            self.exposureFrame.grid(row=7, column=0, sticky='EW')

        if ((self.film_type.get()  == 1) or (self.film_type.get()  == 2)) and not self.reject_check.get():
            self.colourFrame.grid(row=6, column=0, sticky='EW') # Show the colour frame only for colour and slides
        else:
            self.colourFrame.grid_forget()
        
        if self.reject_check.get():
            self.cropFrame.grid_forget()
            self.globFrame.grid_forget()
            self.filmFrame.grid_forget()
            self.dustFrame.grid_forget()
            self.editmenu.entryconfigure('Reset to Default Settings', state=tk.DISABLED)
        else:
            self.cropFrame.grid(row=4, column=0, sticky='EW')
            self.globFrame.grid(row=1, column=0, sticky='EW')
            self.filmFrame.grid(row=3, column=0, sticky='EW')
            self.dustFrame.grid(row=4, column=0, sticky='EW')
            self.editmenu.entryconfigure('Reset to Default Settings', state=tk.NORMAL)
        
        self._update_export_controls()

        film = self.film_type.get()
        for widget in self.clip_widgets:
            widget.hide() if film == 2 else widget.show() # a slide uses a fixed density window, no clip metering
        for key in ('crosstalk_profile', 'crosstalk_strength'):
            self.widgets[key].show() if film == 1 else self.widgets[key].hide()
        
        if self.base_mode.get() and film == 1:
            self.base_clr_lbl.grid(row=1, column=0, sticky=tk.E)
            self.base_rgb_lbl.grid(row=1, column=1, sticky=tk.W)
            self.rgb_display.grid(row=1, column=2, sticky=tk.W)
            self.base_pick_button.grid(row=1, column=3, sticky=tk.W)
            self.base_buttons_frame.grid(row=2, column=0, columnspan=4, sticky='e')
        else:
            self.base_clr_lbl.grid_forget()
            self.base_rgb_lbl.grid_forget()
            self.rgb_display.grid_forget()
            self.base_pick_button.grid_forget()
            self.base_buttons_frame.grid_forget()

    def toggle_paper_settings(self, widget):
        # Shows/hides the paper profile controls based on the toggle's state
        for paper_widget in self.paper_widgets:
            paper_widget.show() if widget.get() else paper_widget.hide()

    def set_flip(self, event=None):
        if self.glob_check.get():
            affected = sum([photo.use_global_settings for photo in self.photos])
            if affected > 1:
                if messagebox.askyesno('Flip Photo', f'Do you want to globally flip {str(affected)} photos?', default='no'):
                    self.global_settings['flip'] = self.flip_check.get()
                    self.changed_global_settings()
                else:
                    self.glob_check.set(False)
                    self.current_photo.use_global_settings = False
            else:
                self.global_settings['flip'] = self.widgets['flip'].get()
                self.changed_global_settings()
        if len(self.photos) == 0:
            return
        self.current_photo.flip = self.widgets['flip'].get()
        self.update_IMG()
        self.unsaved = True
    
    def rot_clockwise(self, event=None):
        if len(self.photos) == 0:
            return
        if self.flip_check.get():
            self.current_photo.rotation -= 1
        else:
            self.current_photo.rotation += 1
        self.update_IMG()
        self.unsaved = True
    
    def rot_counterclockwise(self, event=None):
        if len(self.photos) == 0:
            return
        if self.flip_check.get():
            self.current_photo.rotation += 1
        else:
            self.current_photo.rotation -= 1
        self.update_IMG()
        self.unsaved = True

    def pick_wb(self):
        # Enables the white balance picker and cursor
        if len(self.photos) == 0:
            return
        self.wb_picker_button.state(['pressed']) # keeps the button pressed
        self.result_photo.configure(cursor='tcross') # changes the cursor over the preview to a cross
        self.result_photo_frame.configure(background='red') # highlights the preview image
        self.wb_picker = True # flag to indicate that the wb picker is enabled
        self.frame.set(0, True)

    def click(self, event):
        # Event handler for all clicks on GUI
        event.widget.focus_set() # if clicked anywhere, set focus to that widget
        if self.wb_picker and (event.widget != self.wb_picker_button):
            # Logic for if the white balance picker is selected
            self.wb_picker_button.state(['!pressed']) # reset button
            self.result_photo.configure(cursor='arrow') # reset cursor
            self.result_photo_frame.configure(background=self.master.cget('bg')) # reset frame
            self.wb_picker = False
            if event.widget == self.result_photo:
                x = event.x / event.widget.winfo_width()
                y = event.y / event.widget.winfo_height()
                self.current_photo.set_wb_from_picker(x, y) # set the white balance to neutral at the x, y coordinates of the mouse click
                picked = {key: getattr(self.current_photo, key) for key in ('wb_cyan', 'wb_magenta', 'wb_yellow')}
                for key, value in picked.items():
                    self.widgets[key].set(value)
                self._sync_temperature()

                if self.glob_check.get():
                    affected = sum([photo.use_global_settings for photo in self.photos])
                    if affected > 1:
                        if messagebox.askyesno('White Balance Picker', f'Do you want to apply this white balance adjustment globally to {str(affected)} photos?', default='no'):
                            self.global_settings.update(picked)
                            self.changed_global_settings()
                        else:
                            self.glob_check.set(False)
                            self.current_photo.use_global_settings = False
                    else:
                        self.global_settings.update(picked)
                        self.changed_global_settings()
                self.update_IMG()
                self.unsaved = True
        elif self.base_picker and (event.widget != self.base_pick_button):
            # Logic for if the base picker is selected
            self.base_pick_button.state(['!pressed']) # reset button
            self.process_photo.configure(cursor='arrow') # reset cursor
            self.process_photo_frame.configure(background=self.master.cget('bg')) # reset frane
            self.base_picker = False
            if event.widget == self.process_photo:
                x = event.x / event.widget.winfo_width()
                y = event.y / event.widget.winfo_height()
                self.current_photo.get_base_colour(x,y) # retrieve base colour at the x, y coordinates of the mouse click
                self.set_base_rgb(1)

    def set_base_detect(self, event=None):
        # Switches between auto base colour detect and manual setting
        if self.glob_check.get():
            affected = sum([photo.use_global_settings for photo in self.photos])
            if affected > 1:
                if messagebox.askyesno('Base Colour Changed', f'Do you want change the base colour globally to {str(affected)} photos?', default='no'):
                    self.global_settings['base_detect'] = self.base_mode.get()
                    self.changed_global_settings()
                else:
                    self.glob_check.set(False)
                    self.current_photo.use_global_settings = False
            else:
                self.global_settings['base_detect'] = self.base_mode.get()
                self.changed_global_settings()
        self.update_UI()
        if len(self.photos) == 0:
            return
        self.current_photo.base_detect = self.base_mode.get()
        self.current_photo.process(skip_crop=True)
        self.update_IMG()
        self.unsaved = True

    # logic to get the base rgb value, and configure the GUI accordingly
    def set_base_rgb(self, mode):
        match mode:
            case 0: # set rgb from colour chooser (8-bit values as displayed)
                try:
                    colour = self._base_swatch(self.current_photo.base_rgb)
                except Exception as e:
                    logger.exception(f'Exception: {e}')
                    colour = None
                rgb, _ = colorchooser.askcolor(colour, title='Enter Film Base RGB')
                if rgb is None:
                    return
                rgb = self._swatch_to_linear(rgb)
            case 1: # pick colour from RAW image
                rgb = self.current_photo.base_rgb
            case 2: # pick colour from blank scan
                filename = filedialog.askopenfile(title='Select Blank Film Scan', filetypes=self.allowable_image_filetypes)
                try:
                    filename = filename.name
                except Exception as e: 
                    logger.exception(f'Exception: {e}')
                    return
                blank = RawProcessing(filename, self.default_settings, self.global_settings, self.config_path)
                blank.film_type = 1
                blank.load()
                if blank.FileReadError:
                    messagebox.showerror('Read Blank', 'RAW image could not be read. Verify the integrity of the RAW image.')
                    return
                rgb = blank.blank_base_colour() # linear colour at the 90th percentile brightest pixels
                blank.clear_memory()

        if self.glob_check.get():
            self.global_settings['base_rgb'] = rgb
            self.changed_global_settings()
        self._set_base_display(rgb)
        if len(self.photos) == 0:
            return
        self.current_photo.base_rgb = rgb
        self.current_photo.process(skip_crop=True)
        self.update_IMG()
        self.unsaved = True

    # --- film base helpers: the photo stores a linear RGB triple, the GUI shows its display-encoded swatch
    def _display_lut(self):
        photo = getattr(self, 'current_photo', None)
        lut = getattr(photo, '_display_lut', None)
        if lut is None:
            from conversion.decode import build_display_lut
            lut = build_display_lut(1.0)
        return lut

    def _base_swatch(self, rgb):
        if rgb is None:
            return (128, 128, 128)
        lut = self._display_lut()
        idx = np.clip(np.rint(np.asarray(rgb, dtype=np.float64)[:3] * 65535), 0, 65535).astype(np.int64)
        return tuple(int(lut[i]) for i in idx)

    def _swatch_to_linear(self, rgb):
        lut = self._display_lut()
        return tuple(float(np.searchsorted(lut, int(round(c)), side='left')) / 65535.0 for c in rgb)

    def _set_base_display(self, rgb):
        swatch = self._base_swatch(rgb)
        self.base_rgb.set(str(swatch) if rgb is not None else 'Auto')
        self.rgb_display.configure(bg=self._from_rgb(swatch))

    # --- colour temperature is derived from magenta/yellow filtration, never stored
    def _sync_temperature(self):
        try:
            kelvin = wb_to_kelvin(self.widgets['wb_magenta'].get(), self.widgets['wb_yellow'].get())
            self.temperature.set(int(round(min(max(kelvin, 3000), 12000))))
        except Exception as e:
            logger.info(f'Could not derive temperature: {e}')

    def _set_filtration(self, values):
        # writes magenta/yellow to the widgets, the current photo and (when synced) the global settings
        for key, value in values.items():
            self.widgets[key].set(value)
            if hasattr(self, 'current_photo'):
                setattr(self.current_photo, key, value)
            if self.glob_check.get():
                self.global_settings[key] = value
        if self.glob_check.get():
            self.changed_global_settings()

    def temperature_changed(self, widget):
        m, y = kelvin_to_wb(widget.get(), self.widgets['wb_magenta'].get(), self.widgets['wb_yellow'].get())
        self._set_filtration({'wb_magenta': m, 'wb_yellow': y})
        if len(self.photos) == 0:
            return
        self.current_photo.process(skip_crop=True)
        self.update_IMG()
        self.unsaved = True

    def filtration_changed(self, widget):
        self.widget_changed(widget, 'skip crop')
        self._sync_temperature()

    # --- film type switches reset Cast Removal / autos to the mode's defaults and refresh the paper list
    def film_type_changed(self, widget):
        film = widget.get()
        current = {key: self.widgets[key].get() for key in ('cast_removal', 'auto_density', 'auto_grade')}
        for key, value in mode_defaults(film, current).items():
            if value != current[key]:
                self.widgets[key].set(value)
                if self.glob_check.get():
                    self.global_settings[key] = value
                if hasattr(self, 'current_photo'):
                    setattr(self.current_photo, key, value)
        if hasattr(self, 'current_photo'):
            self.current_photo.film_type = film
        self._refresh_papers(film)
        self.widget_changed(widget)

    def _refresh_papers(self, film=None):
        film = self.film_type.get() if film is None else film
        mode = {0: ProcessMode.BW, 2: ProcessMode.E6}.get(film, ProcessMode.C41)
        profiles = profiles_for_mode(mode)
        keys = [key for key, _ in profiles]
        labels = [getattr(profile, 'label', None) or getattr(profile, 'name', None) or key for key, profile in profiles]
        self.paper_keys[:] = keys
        self.paper_combo.values = labels
        self.paper_combo.combobox.configure(values=labels)
        wanted = getattr(getattr(self, 'current_photo', None), 'paper_profile', 'neutral')
        if wanted not in keys:
            wanted = keys[0]
            if hasattr(self, 'current_photo'):
                self.current_photo.paper_profile = wanted
        self.paper_combo.set(wanted)

    def _update_effective_grade(self):
        grade = self.current_photo.effective_grade() if hasattr(self, 'current_photo') else None
        self.effective_grade_lbl.configure(text='' if grade is None else f'Effective grade R{grade:.0f}')

    def set_base(self):
        # enables the base colour picker
        if len(self.photos) == 0:
            return
        self.base_pick_button.state(['pressed']) # keeps the button pressed
        self.process_photo.configure(cursor='tcross') # changes the cursor over the preview to a cross
        self.photo_process_Combo.current(0) # display the RAW photo
        self.process_photo_frame.configure(background='red') # highlights the raw photo
        self.base_picker = True # flag to indicate that the wb picker is enabled
        self.update_IMG()
    
    def _start_individual_export(self):
        # Start export in separate thread to keep UI responsive
        if len([photo for photo in self.photos if not photo.reject]) == 0:
            return

        self.export_thread = threading.Thread(target=self.export_individual, daemon=True)
        self.export_thread.start()
    
    def export_individual(self):
        # Exports only photo that is currently visible
        if len(self.photos) == 0:
            return
        
        self.show_progress() # display progress bar

        self._set_individual_export_controls_enabled(False)
        
        self.update_progress(20, 'Processing...') # Arbitrary progress display
        self._load_photo(self.current_photo, full_res=True)
        self.current_photo.process(True, for_export=True)
        self.update_progress(99, self._individual_export_progress_message())
        
        self._export_photo(
            self.current_photo,
            self._individual_export_filename(),
        )
        self._set_individual_export_controls_enabled(True)
        self.hide_progress() # hide the progress bar
        self._after_individual_export()
    
    # Defines how to show and hide the progress bar
    def show_progress(self, message=''):
        self.progressFrame.grid(row=10, column=0, sticky='NEW')
        self.update_progress(0, message)
    
    def hide_progress(self):
        self.progressFrame.grid_forget()
    
    def update_progress(self, percentage, message=''):
        # takes number between 0 and 100 to display progress, with optional parameter to display message
        self.progress.set(percentage)
        self.progress_percentage.configure(text=f'{str(round(percentage))}%')
        self.progress_msg.configure(text=message)
        self.master.update()

    def key_handler(self, event):
        # Maps various keybinds to different functions
        if not hasattr(self, 'previous_view'):
            self.previous_view = 0  # Default to RAW

        match event.keysym:
            case 'Right':
                self.next()
            case 'Left':
                self.previous()
            case 'r':
                self.rot_clockwise()
            case 'R':
                self.rot_counterclockwise()
            case 'q':
                self.previous_view = 0
                self.photo_process_Combo.current(0) # RAW
                self.update_IMG()
            case 'w':
                self.previous_view = 1
                self.photo_process_Combo.current(1) # Threshold
                self.update_IMG()
            case 'e':
                self.previous_view = 2
                self.photo_process_Combo.current(2) # Contours
                self.update_IMG()
            case 'h':
                self.previous_view = 3
                self.photo_process_Combo.current(3) # Histogram
                self.update_IMG()
            case 'f':
                current_view = self.photo_process_Combo.current()
                if current_view == 4:
                    self.photo_process_Combo.current(self.previous_view) # Return to previous
                else:
                    self.photo_process_Combo.current(4) # Full Preview
                self.update_IMG()

    def set_tooltip(self, widget, text, delay=500):
        # Adds a tooltip to a provided widget showing the provided text when the mouse is hovering over the widget
        tooltip = tk.Label(self.master, text=f"{text}", bg="white", relief="solid", borderwidth=1, anchor="w", justify="left", padx=3, pady=1)
        tooltip_timer = None
        def show_tooltip():
            tooltip.place(x=widget.winfo_rootx() - self.master.winfo_rootx() + 10, 
                        y=widget.winfo_rooty() - self.master.winfo_rooty() + 30)
        def schedule_tooltip(event):
            nonlocal tooltip_timer
            tooltip_timer = self.master.after(delay, show_tooltip)
        def cancel_tooltip(event):
            nonlocal tooltip_timer
            if tooltip_timer:
                self.master.after_cancel(tooltip_timer)
                tooltip_timer = None
            tooltip.place_forget()
        widget.bind("<Enter>", schedule_tooltip)
        widget.bind("<Leave>", cancel_tooltip)
        
    def on_closing(self):
        # Behaviour/cleanup at closing
        if len(self.photos) > 0:
            if self.ask_save_settings() is None: # if "Cancel" is pressed, do nothing
                return
        if hasattr(self, 'export_thread') and self.export_thread.is_alive(): # check if the export thread is still alive
            if messagebox.askyesno(title='Export in Progress', icon='warning', message='Export is still in progress. Do you really want to quit?', default='no'):
                try:
                    self.pool.terminate()
                except Exception as e:
                    logger.exception(f'Exception: {e}') 
            else:
                return
        self.master.destroy() # quit program

    def ask_save_settings(self):
        # dialog to ask if settings are to be saved. If yes, saves settings and returns True. No returns False. Cancel returns None.
        if self.unsaved:
            result = messagebox.askyesnocancel('Unsaved Changes', 'Do you want to save the changes you made to this batch of photos?')
        else:
            return False
        if result:
            self.save_settings()
        return result
    
    def save_settings(self, event=None):
        # loops through all the photos and applies global settings where needed, then saves the settings to disk
        for photo in self.photos:
            if photo.use_global_settings:
                self.apply_settings(photo, self.global_settings) # apply most current settings before saving
            photo.save_settings()
        self.unsaved = False

    def copy_settings(self, event=None):
        # copies the current photo's settings
        if len(self.photos) == 0:
            return
        self.copied_settings = {}
        for key in self.default_settings:
            self.copied_settings[key] = getattr(self.current_photo, key)
        self.editmenu.entryconfig('Paste Settings', state=tk.NORMAL)
        
    def paste_settings(self, event=None):
        # pastes the copied settings if self.copy_settings() has been called
        # if pasted into a photo with global settings applied, asks if the user wants to apply to all photos with global settings
        if hasattr(self, 'copied_settings') and hasattr(self, 'current_photo'):
            self.apply_settings(self.current_photo, self.copied_settings)
            if self.current_photo.use_global_settings:
                affected = sum([photo.use_global_settings for photo in self.photos])
                if affected > 1:
                    if messagebox.askyesno('Paste Settings', f'Do you want to paste the settings globally to {str(affected)} photos?'):
                        self.global_settings = self.copied_settings
                        self.changed_global_settings()
                    else:
                        self.glob_check.set(False)
                        self.current_photo.use_global_settings = False
                else:
                    self.global_settings = self.copied_settings
            self.change_settings()
            self.current_photo.process()
            self.update_IMG()
            self.unsaved = True
    
    @staticmethod
    def _from_rgb(rgb):
        # translates an rgb tuple of int to a tkinter friendly color code
        return '#%02x%02x%02x' % rgb