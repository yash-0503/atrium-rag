import httpx
import time

class IngestClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def ingest(self, file_path: str, doc_id: str = None) -> dict:
        """Blocking call wrapper around async ingest.

        Uses `/ingest_single` (form field `file`, singular) rather than
        `/ingest` (form field `files`, plural) — the multi-file endpoint
        answers with `{"jobs": [...]}`, not a bare `job_id`.
        """
        with open(file_path, "rb") as f:
            data = {"doc_id": doc_id} if doc_id else {}
            resp = httpx.post(f"{self.base_url}/ingest_single", files={"file": f}, data=data)
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            
        while True:
            status_resp = httpx.get(f"{self.base_url}/jobs/{job_id}").json()
            if status_resp["status"] == "complete":
                return status_resp
            if status_resp["status"] == "failed":
                raise Exception(status_resp["error"])
            time.sleep(2)