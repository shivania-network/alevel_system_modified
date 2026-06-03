"""Process counseling hero image: transparent bg, blue graduate gown, emphasize professional."""
from PIL import Image
from collections import deque

SRC = r"C:\Users\Shiva\.cursor\projects\c-Users-Shiva-OneDrive-Desktop-alevel-system-final\assets\c__Users_Shiva_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_img-f0e18273-059a-4b55-96e9-65b3e7a8baa5.png"
OUT = r"c:\Users\Shiva\OneDrive\Desktop\alevel_system_final\alevel_system\static\img\hero-counseling.png"


def is_bg(r, g, b):
    if r >= 248 and g >= 248 and b >= 248:
        return True
    if r >= 235 and g >= 240 and b >= 245:
        return True
    if r >= 210 and g >= 225 and b >= 240:
        return True
    if r >= 195 and g >= 215 and b >= 235 and abs(r - g) < 30:
        return True
    return False


def remove_background(img):
    w, h = img.size
    px = img.load()
    seen = [[False] * w for _ in range(h)]
    q = deque()

    def seed(x, y):
        if 0 <= x < w and 0 <= y < h and not seen[y][x] and is_bg(*px[x, y][:3]):
            seen[y][x] = True
            q.append((x, y))

    for x in range(w):
        seed(x, 0)
        seed(x, h - 1)
    for y in range(h):
        seed(0, y)
        seed(w - 1, y)

    while q:
        x, y = q.popleft()
        r, g, b, a = px[x, y]
        px[x, y] = (r, g, b, 0)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx]:
                if is_bg(*px[nx, ny][:3]):
                    seen[ny][nx] = True
                    q.append((nx, ny))
    return img


def gown_to_blue(img):
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w // 2 + 40):  # left side — graduate
            r, g, b, a = px[x, y]
            if a < 20:
                continue
            if r < 75 and g < 75 and b < 85 and max(r, g, b) - min(r, g, b) < 25:
                px[x, y] = (30, 58, 95, a)
            elif r < 95 and g < 95 and b < 110 and r < g + 15:
                px[x, y] = (45, 90, 142, a)
    return img


def emphasize_professional(img):
    w, h = img.size
    left = img.crop((0, 0, int(w * 0.42), h))
    right = img.crop((int(w * 0.32), 0, w, h))
    lw, lh = left.size
    rw, rh = right.size
    left = left.resize((int(lw * 0.88), int(lh * 0.88)), Image.LANCZOS)
    right = right.resize((int(rw * 1.18), int(rh * 1.18)), Image.LANCZOS)
    canvas_w = int(w * 0.78)
    canvas_h = int(h * 0.78)
    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    out.paste(left, (0, max(0, (canvas_h - left.height) // 2)), left)
    rx = max(0, canvas_w - right.width + 8)
    out.paste(right, (rx, max(0, (canvas_h - right.height) // 2)), right)
    return out


def main():
    img = Image.open(SRC).convert("RGBA")
    img = remove_background(img)
    img = gown_to_blue(img)
    img = emphasize_professional(img)
    img.save(OUT, "PNG")
    print("saved", OUT, img.size)


if __name__ == "__main__":
    main()
