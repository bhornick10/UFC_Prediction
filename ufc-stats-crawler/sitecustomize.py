# Shim for old Scrapy on modern OpenSSL (SSLv3 removed).
try:
    from OpenSSL import SSL as _SSL
    if not hasattr(_SSL, "SSLv3_METHOD"):
        _SSL.SSLv3_METHOD = object()  # stub so import in scrapy.tls succeeds
except Exception:
    pass
