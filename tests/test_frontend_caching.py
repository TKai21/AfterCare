async def test_frontend_files_are_revalidated(client):
    """Unbundled frontend files keep their names across versions: without
    no-cache a browser keeps running the old JavaScript after an update."""
    for path in ("/", "/app", "/doctor", "/static/review.js", "/static/app.css", "/static/vendor/pdfjs/pdf.min.mjs"):
        resp = await client.get(path)
        assert resp.status_code == 200, path
        assert resp.headers["cache-control"] == "no-cache", path
        assert resp.headers.get("etag") or path in ("/", "/app", "/doctor"), path


async def test_unchanged_static_file_is_a_304(client):
    first = await client.get("/static/review.js")
    again = await client.get("/static/review.js", headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
