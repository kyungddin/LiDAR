from ouster.sdk import open_source

HOST = "192.168.0.49"
source = open_source(HOST)

print(f"총 리턴 수: {len(source.metadata)}")

for i, meta in enumerate(source.metadata):
    print(f"\n📦 [Return {i}] Metadata 요약:")
    print(f"  - 이름: {meta.prod_line} {meta.mode}")
    print(f"  - 리턴 모드: {meta.return_mode}")
    print(f"  - 해상도: {meta.format.columns_per_frame} x {meta.format.pixels_per_column}")
    print(f"  - 필드 목록: {list(meta.fields)}")