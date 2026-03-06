"""
全APIエンドポイントの動作確認
サーバー起動中に別ターミナルで実行:
  python scripts/test_api.py
"""
import io
import time
import requests
from PIL import Image

BASE = "http://localhost:8000"


def test_all():
    # 1. GET /info
    print("=== GET /info ===")
    r = requests.get(f"{BASE}/info")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    info = r.json()
    print(info)

    # 2. GET /images
    print("\n=== GET /images ===")
    r = requests.get(f"{BASE}/images")
    assert r.status_code == 200
    images = r.json()["images"]
    print(f"画像数: {len(images)}")
    if images:
        print(f"最初の3枚: {images[:3]}")

    # 3. GET /images/{id}/download
    if images:
        img_id = images[0]["id"]
        print(f"\n=== GET /images/{img_id}/download ===")
        r = requests.get(f"{BASE}/images/{img_id}/download")
        assert r.status_code == 200
        print(f"Content-Type: {r.headers.get('content-type')}, Size: {len(r.content)} bytes")

    # 4. パストラバーサル検証
    print("\n=== パストラバーサル検証 ===")
    # FastAPIはパスに/を含むとルーティング不一致で404になるため、
    # validate_image_idが機能するケース（..を含むがスラッシュなし）をテスト
    r = requests.get(f"{BASE}/images/..img.png/download")
    print(f"..img.png → Status: {r.status_code} (expected 400)")
    assert r.status_code == 400
    r = requests.get(f"{BASE}/images/bad%00name.png/download")
    print(f"bad%%00name.png → Status: {r.status_code} (expected 400)")
    assert r.status_code == 400

    # 5. POST /infer/{id}
    if images:
        img_id = images[0]["id"]
        print(f"\n=== POST /infer/{img_id} ===")
        start = time.time()
        r = requests.post(f"{BASE}/infer/{img_id}")
        elapsed = time.time() - start
        if r.status_code == 200:
            print(f"マスクサイズ: {len(r.content)} bytes, レスポンス時間: {elapsed:.2f}秒")
            if elapsed < 2.0:
                print(f"  ✓ NFR-1 合格 (< 2秒)")
            else:
                print(f"  ✗ NFR-1 不合格 (>= 2秒)")
            with open("test_prediction.png", "wb") as f:
                f.write(r.content)
            print("  → test_prediction.png に保存")
        else:
            print(f"Status: {r.status_code}, {r.json()}")

    # 6. GET /next
    print("\n=== GET /next ===")
    r = requests.get(f"{BASE}/next")
    assert r.status_code == 200
    print(r.json())

    # 7. GET /next?strategy=sequential
    print("\n=== GET /next?strategy=sequential ===")
    r = requests.get(f"{BASE}/next", params={"strategy": "sequential"})
    assert r.status_code == 200
    print(r.json())

    # 8. PUT /submit/{id} (ダミーマスク送信)
    if images:
        img_id = images[0]["id"]
        print(f"\n=== PUT /submit/{img_id} ===")
        dummy = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        buf = io.BytesIO()
        dummy.save(buf, format="PNG")
        buf.seek(0)
        r = requests.put(
            f"{BASE}/submit/{img_id}",
            files={"file": (img_id, buf.getvalue(), "image/png")},
        )
        assert r.status_code == 200
        print(r.json())

    # 9. GET /labels/{id}/download
    if images:
        img_id = images[0]["id"]
        print(f"\n=== GET /labels/{img_id}/download ===")
        r = requests.get(f"{BASE}/labels/{img_id}/download")
        print(f"Status: {r.status_code}, Size: {len(r.content)} bytes")

    # 10. GET /status
    print("\n=== GET /status ===")
    r = requests.get(f"{BASE}/status")
    assert r.status_code == 200
    print(r.json())

    # 11. POST /train (epochを3に絞ってテスト)
    print("\n=== POST /train (3 epochs) ===")
    r = requests.post(f"{BASE}/train", params={"max_epochs": 3})
    print(f"Status: {r.status_code}, {r.json()}")

    if r.status_code == 200:
        # 12. 学習中に再度トレーニングリクエスト (409確認)
        print("\n=== POST /train (学習中の二重リクエスト) ===")
        r2 = requests.post(f"{BASE}/train", params={"max_epochs": 3})
        print(f"Status: {r2.status_code} (expected 409), {r2.json()}")

        # 13. ポーリングで学習完了を待つ
        print("\n=== 学習ポーリング ===")
        for _ in range(60):
            time.sleep(3)
            r = requests.get(f"{BASE}/status")
            status = r.json()
            print(
                f"  {status['status']} epoch={status['epoch']}/{status['max_epochs']} "
                f"dice={status['best_dice']:.4f}"
            )
            if status["status"] != "running":
                break

    # 14. 存在しない画像のテスト
    print("\n=== 404テスト ===")
    r = requests.get(f"{BASE}/images/nonexistent.png/download")
    print(f"GET nonexistent image → Status: {r.status_code} (expected 404)")
    assert r.status_code == 404

    r = requests.get(f"{BASE}/labels/nonexistent.png/download")
    print(f"GET nonexistent label → Status: {r.status_code} (expected 404)")
    assert r.status_code == 404

    print("\n=== 全テスト完了 ===")


if __name__ == "__main__":
    test_all()
