from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_size: int):
        super().__init__(app)
        self.max_body_size = int(max_body_size)

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > self.max_body_size:
                    return JSONResponse({"error": "payload_too_large"}, status_code=413)
            except Exception:
                # If content-length is malformed, don't block here; downstream will handle
                pass
        return await call_next(request)
