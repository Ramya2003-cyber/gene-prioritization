// Where the FastAPI backend runs. Change API_BASE after deploying the backend.
window.APP_CONFIG = {
  API_BASE: location.hostname === "localhost" || location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://gene-prioritizer-api.onrender.com",
  REPO_URL: "https://github.com/Ramya2003-cyber/gene-prioritization",
};
