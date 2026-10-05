import unittest
from unittest.mock import patch

from sum_ui.browse import browse_file, browse_folder


class _FakeRoot:
    def __init__(self):
        self.calls = []

    def withdraw(self):
        self.calls.append(("withdraw",))

    def attributes(self, *args):
        self.calls.append(("attributes", *args))

    def update(self):
        self.calls.append(("update",))

    def destroy(self):
        self.calls.append(("destroy",))


class BrowseDialogTests(unittest.TestCase):
    def test_folder_picker_uses_native_tk_dialog(self):
        root = _FakeRoot()
        with patch("tkinter.Tk", return_value=root), \
             patch("tkinter.filedialog.askdirectory", return_value=r"D:\data") as dialog:
            result = browse_folder("Pick folder")
        self.assertEqual(result, r"D:\data")
        self.assertEqual(dialog.call_args.kwargs["title"], "Pick folder")
        self.assertIs(dialog.call_args.kwargs["parent"], root)
        self.assertIn(("destroy",), root.calls)

    def test_file_picker_passes_filters_to_native_tk_dialog(self):
        root = _FakeRoot()
        filters = [("JSON", "*.json")]
        with patch("tkinter.Tk", return_value=root), \
             patch("tkinter.filedialog.askopenfilename", return_value=r"D:\data\labels.json") as dialog:
            result = browse_file("Pick JSON", filters)
        self.assertEqual(result, r"D:\data\labels.json")
        self.assertEqual(dialog.call_args.kwargs["filetypes"], filters)
        self.assertIs(dialog.call_args.kwargs["parent"], root)

    def test_dialog_errors_return_empty_path(self):
        with patch("tkinter.Tk", side_effect=RuntimeError("no desktop")):
            self.assertEqual(browse_folder(), "")
            self.assertEqual(browse_file(), "")


if __name__ == "__main__":
    unittest.main()
