import glob
import logging
import multiprocessing
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import psutil

from BaseGUI import BaseGUI
from RawProcessing import RawProcessing


logger = logging.getLogger(__name__)


class GUI(BaseGUI):
    def __init__(self, master):
        self.destination_folder = ''
        super().__init__(master)

    def _build_file_menu_controls(self):
        self.filemenu.add_command(label='Import...', command=self.import_photos)

    def _build_import_controls(self, import_frame):
        import_subframe_1 = ttk.Frame(import_frame)
        import_subframe_1.pack(fill='x')
        ttk.Label(import_subframe_1, text='RAW File:').pack(side=tk.LEFT)
        self.photoCombo = ttk.Combobox(import_subframe_1, state='readonly')
        self.photoCombo.bind('<<ComboboxSelected>>', self.load_IMG)
        self.photoCombo.pack(side=tk.LEFT, padx=2)
        self.import_button = ttk.Button(
            import_subframe_1,
            text='Import...',
            command=self.import_photos,
            width=8,
        )
        self.import_button.pack(side=tk.LEFT, padx=2)

        import_subframe_2 = ttk.Frame(import_frame)
        import_subframe_2.pack(fill='x')
        self.prevButton = ttk.Button(
            import_subframe_2,
            text='< Previous Photo',
            width=20,
            command=self.previous,
        )
        self.prevButton.pack(side=tk.LEFT, padx=2, pady=5)
        self.set_tooltip(self.prevButton, '(Left Arrow)')
        self.nextButton = ttk.Button(
            import_subframe_2,
            text='Next Photo >',
            width=20,
            command=self.next,
        )
        self.nextButton.pack(side=tk.LEFT, padx=2, pady=5)
        self.set_tooltip(self.nextButton, '(Right Arrow)')

    def _build_export_controls(self, export_frame):
        self.export_destination_heading = ttk.Label(
            export_frame,
            text='Output Destination Folder:',
            anchor='w',
        )
        self.export_destination_heading.pack(fill='x')
        self.destination_folder_text = tk.StringVar()
        self.destination_folder_text.set('No Destination Folder Specified')
        self.export_destination_lbl = ttk.Label(
            export_frame,
            textvariable=self.destination_folder_text,
            anchor='w',
            font=('Segoe UI', 9, 'italic'),
        )
        self.export_destination_lbl.pack(fill='x')
        self.export_destination_lbl.bind(
            '<Configure>',
            lambda event: self.export_destination_lbl.config(
                wraplength=self.export_destination_lbl.winfo_width()
            ),
        )
        self.select_folder_button = ttk.Button(
            export_frame,
            text='Select Folder',
            command=self.select_folder,
        )
        self.select_folder_button.pack(side=tk.LEFT, padx=2, pady=5)
        self.current_photo_button = ttk.Button(
            export_frame,
            text='Export Current Photo',
            command=self.export,
            state=tk.DISABLED,
        )
        self.current_photo_button.pack(side=tk.LEFT, padx=2, pady=5)
        self.all_photo_button = ttk.Button(
            export_frame,
            text='Export All Photos',
            command=lambda: self.export(len(self.photos)),
            state=tk.DISABLED,
        )
        self.all_photo_button.pack(side=tk.LEFT, padx=2, pady=5)
        self.abort_button = ttk.Button(
            export_frame,
            text='Abort Export',
            command=self.abort,
        )

    def import_photos(self):
        if len(self.photos) > 0 and self.ask_save_settings() is None:
            return
        if hasattr(self, 'export_thread') and self.export_thread.is_alive():
            return

        filenames = filedialog.askopenfilenames(
            title='Select RAW File(s)',
            filetypes=self.allowable_image_filetypes,
        )
        if filenames:
            self.import_from_filenames(filenames)

    def load_all_from_path(self, pathname):
        print(f'Loading path: {pathname}')
        extensions = self.allowable_image_filetypes[0][1].split()
        extensions.extend(self.allowable_image_filetypes[1][1].split())
        files = []
        for extension in extensions:
            files.extend(glob.glob(os.path.join(pathname, extension)))
        files = remove_duplicate_strings(files)
        if files:
            self.import_from_filenames(files)

    def _current_photo_index(self):
        return self.photoCombo.current()

    def _set_import_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.import_button.configure(state=state)
        self.filemenu.entryconfigure('Import...', state=state)

    def _set_imported_photo_names(self, photo_names):
        self.photoCombo.configure(values=photo_names)
        self.photoCombo.current(0)

    def _update_photo_navigation(self):
        index = self.photoCombo.current()
        self.prevButton.configure(
            state=tk.DISABLED if index <= 0 else tk.NORMAL
        )
        self.nextButton.configure(
            state=(
                tk.DISABLED
                if index + 1 >= len(self.photos)
                else tk.NORMAL
            )
        )

    def _update_export_controls(self):
        current_state = (
            tk.DISABLED
            if self.reject_check.get() or len(self.photos) == 0
            else tk.NORMAL
        )
        self.current_photo_button.configure(state=current_state)
        all_state = (
            tk.DISABLED
            if not any(not photo.reject for photo in self.photos)
            else tk.NORMAL
        )
        self.all_photo_button.configure(state=all_state)

    def _individual_export_filename(self):
        return os.path.join(
            self.destination_folder,
            os.path.splitext(str(self.current_photo))[0],
        )

    def _set_individual_export_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.current_photo_button.configure(state=state)
        self.all_photo_button.configure(state=state)
        self._set_import_controls_enabled(enabled)

    def previous(self):
        if len(self.photos) == 0:
            return
        index = self.photoCombo.current()
        if index > 0:
            self.photoCombo.current(index - 1)
            self.load_IMG()
        self._update_photo_navigation()

    def next(self):
        if len(self.photos) == 0:
            return
        index = self.photoCombo.current()
        if index < len(self.photos) - 1:
            self.photoCombo.current(index + 1)
            self.load_IMG()
        self._update_photo_navigation()

    def set_disable_buttons(self):
        self._update_photo_navigation()

    def select_folder(self):
        destination_folder = filedialog.askdirectory() + '/'
        if len(destination_folder) > 1:
            self.set_destination_folder(destination_folder)

    def set_destination_folder(self, destination_folder):
        self.destination_folder = destination_folder
        self.destination_folder_text.set(destination_folder)

    def export(self, n_photos=1):
        if not any(not photo.reject for photo in self.photos):
            return
        if not self.destination_folder:
            self.destination_folder = 'export'
        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)
            logger.info(f'Creating {self.destination_folder} folder')

        if n_photos == 1:
            self._start_individual_export()
        else:
            self.export_thread = threading.Thread(
                target=self.export_multiple,
                daemon=True,
            )
            self.export_thread.start()

    def export_multiple(self):
        if len(self.photos) == 0:
            return
        self.show_progress('Applying photo settings...')
        self.current_photo_button.configure(state=tk.DISABLED)
        self.all_photo_button.pack_forget()
        self.abort_button.pack(side=tk.LEFT, padx=2, pady=5)
        self._set_import_controls_enabled(False)

        inputs = []
        allocated = 0
        has_alloc = 0
        with multiprocessing.Manager() as manager:
            self.terminate = manager.Event()
            for photo in self.photos:
                if photo.reject:
                    continue
                if photo.use_global_settings:
                    self.apply_settings(photo, self.global_settings)
                filename = os.path.join(
                    self.destination_folder,
                    os.path.splitext(str(photo))[0],
                )
                inputs.append(
                    (
                        photo,
                        filename,
                        self.terminate,
                        RawProcessing.class_parameters,
                    )
                )
                if hasattr(photo, 'memory_alloc'):
                    allocated += photo.memory_alloc
                    has_alloc += 1

            if self.advanced_settings['max_processors_override'] != 0:
                max_processors = self.advanced_settings[
                    'max_processors_override'
                ]
            else:
                available = psutil.virtual_memory()[1]
                allocated = allocated / has_alloc
                max_processors = round(available / allocated)
            processes = max(
                min(
                    max_processors,
                    multiprocessing.cpu_count(),
                    len(inputs),
                ),
                1,
            )

            self.update_progress(
                20,
                f'Allocating {processes} processor(s) for export...',
            )
            with multiprocessing.Pool(processes) as self.pool:
                errors = []
                for index, result in enumerate(
                    self.pool.imap(self.export_async, inputs),
                    1,
                ):
                    if self.terminate.is_set():
                        self.pool.terminate()
                        break
                    if result:
                        errors.append(result)
                        logger.exception(f'Exception: {result}')
                    update_message = (
                        f'Exported {index} of {len(inputs)} photos.'
                    )
                    self.update_progress(
                        index / len(inputs) * 80 + 19.99,
                        update_message,
                    )

        if errors and not self.terminate.is_set():
            errors_display = 'Details:'
            for index, error in enumerate(errors, 1):
                errors_display += f'\n {index}. {error}'
            messagebox.showerror(
                f'Export Error: {len(errors)}) export(s) failed.\n'
                + errors_display
            )

        self.current_photo_button.configure(state=tk.NORMAL)
        self.abort_button.pack_forget()
        self.all_photo_button.pack(side=tk.LEFT, padx=2, pady=5)
        self._set_import_controls_enabled(True)
        self.hide_progress()

    def abort(self):
        try:
            self.terminate.set()
        except Exception:
            pass

    @staticmethod
    def export_async(inputs):
        photo, filename, terminate, class_parameters = inputs
        RawProcessing.class_parameters = class_parameters
        for _ in range(5):
            try:
                if terminate.is_set():
                    return
                photo.load(True)
                if photo.FileReadError:
                    raise Exception('File could not be read')
                if terminate.is_set():
                    return
                photo.process(True)
            except Exception as exception:
                error = exception
            else:
                if terminate.is_set():
                    return
                photo.export(filename)
                photo.clear_memory()
                return False
        return error


def remove_duplicate_strings(strings):
    seen = set()
    result = []
    for value in strings:
        lower_value = value.lower()
        if lower_value not in seen:
            seen.add(lower_value)
            result.append(value)
    return result
