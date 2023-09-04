
CORNER_DETECTION_ERROR = 'CORNER_DETECTION_ERROR'
PIECE_DETECTION_ERROR = 'PIECE_DETECTION_ERROR'

ANALYSIS_ERROR_TYPES = [
    CORNER_DETECTION_ERROR,
    PIECE_DETECTION_ERROR,
]


class AnalysisError(Exception):
    def __init__(self, message, err_type, data=None):
        super().__init__(message)
        if err_type not in ANALYSIS_ERROR_TYPES:
            print('[WARNING] Invalid error type:', err_type)
        self.type = err_type
        self.data = data


class CornerDetectionError(AnalysisError):
    def __init__(self, message, data=None):
        super().__init__(message, CORNER_DETECTION_ERROR, data)


class PieceDetectionError(AnalysisError):
    def __init__(self, message, data=None):
        super().__init__(message, PIECE_DETECTION_ERROR, data)