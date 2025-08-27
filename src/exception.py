import sys  # For system-specific details

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        """
        Builds a detailed error message including filename and line number.
        """
        _, _, exc_tb = sys.exc_info()

        # Sometimes exc_tb can be None, handle that case
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in {file_name}, line {line_number}: {error_message}"
        else:
            return f"{error_message} (no traceback available)"

    def __str__(self):
        return self.error_message
