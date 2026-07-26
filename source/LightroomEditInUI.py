import os
import tkinter as tk
from tkinter import ttk

from BaseGUI import BaseGUI
from RawProcessing import RawProcessing


class LightroomEditInUI(BaseGUI):
    def __init__(self, master, lightroom_path):
        self.lightroom_path = lightroom_path
        RawProcessing.class_parameters['filetype'] = 'TIFF'
        super().__init__(master)
        RawProcessing.class_parameters['filetype'] = 'TIFF'
        self.master.title(
            f'Film Scan Converter — {os.path.basename(lightroom_path)}'
        )

    def _build_file_menu_controls(self):
        pass

    def _build_import_controls(self, import_frame):
        import_subframe = ttk.Frame(import_frame)
        import_subframe.pack(fill='x')
        ttk.Label(import_subframe, text='File:').pack(side=tk.LEFT)
        self.lightroom_filename_lbl = ttk.Label(
            import_subframe,
            text=self.lightroom_path,
            font=('Segoe UI', 9),
        )
        self.lightroom_filename_lbl.pack(
            side=tk.LEFT,
            padx=2,
            fill='x',
            expand=True,
        )

    def _build_export_controls(self, export_frame):
        self.reject_check.hide()
        self.export_filetype_widget.hide()
        self.current_photo_button = ttk.Button(
            export_frame,
            text='Save and Return to Lightroom',
            command=self.export,
            state=tk.DISABLED,
        )
        self.current_photo_button.pack(side=tk.LEFT, padx=2, pady=5)

    def _current_photo_index(self):
        return 0

    def _set_import_controls_enabled(self, enabled):
        pass

    def _set_imported_photo_names(self, photo_names):
        pass

    def _update_photo_navigation(self):
        pass

    def _update_export_controls(self):
        state = (
            tk.DISABLED
            if self.reject_check.get() or len(self.photos) == 0
            else tk.NORMAL
        )
        self.current_photo_button.configure(state=state)

    def _individual_export_filename(self):
        return self.lightroom_path

    def _set_individual_export_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.current_photo_button.configure(state=state)

    def _individual_export_progress_message(self):
        return 'Saving photo...'

    def _after_individual_export(self):
        self.master.destroy()

    def _load_photo(self, photo, full_res=False):
        photo.load_lightroom_tiff()

    def _export_photo(self, photo, filename):
        photo.export_lightroom_tiff(filename)

    def import_lightroom_edit_in(self, path):
        self.lightroom_path = path
        self.lightroom_filename_lbl.configure(text=path)
        self.import_from_filenames((path,))

    def export(self):
        self._start_individual_export()
