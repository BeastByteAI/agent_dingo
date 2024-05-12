def sha256_to_uuid(sha256_hash: str) -> str:
    short_hash = sha256_hash[:32]
    formatted_uuid = f"{short_hash[:8]}-{short_hash[8:12]}-{short_hash[12:16]}-{short_hash[16:20]}-{short_hash[20:32]}"
    return formatted_uuid
