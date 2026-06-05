# Gartic OpenCV Drawer

用 OpenCV + PyAutoGUI 將圖片轉成 Gartic Phone 可繪製的線條、色塊或 RGB 繪圖動作。

> 這是個人輔助工具。使用時請尊重遊戲規則、房間規範與其他玩家體驗。

## 功能

- 自動偵測 Gartic 畫布、固定 18 色色盤、筆刷按鈕與 RGB 面板位置
- 圖片預覽，先確認轉換效果再開始畫
- 支援多種繪製模式：
  - `Smart Line Art` 整合線稿
  - `Palette Color` 固定色盤上色
  - `Custom RGB` 自訂 RGB 色
  - `SBR` 筆觸渲染
- 支援 `Esc` / Stop 按鈕緊急停止
- 可調整 CPS、筆刷大小、圖片縮放、線條細節、移動速度、RGB 顏色數
- `Eye Detail` 可在色盤 / RGB 模式最後補強眼睛細節，避免被底色吃掉

## 安裝

建議使用 Python 3.10 以上。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 執行

```powershell
python gartic_auto_drawer.py
```

## 使用方式

1. 開啟 Gartic Phone 繪圖頁面，確認畫布、色盤、筆刷列都看得到。
2. 執行程式並點 `Load Image` 載入圖片。
3. 點 `Auto Detect` 自動偵測畫布、色盤與筆刷位置。
4. 選擇模式並點 `Preview` 先看效果。
5. 點 `Draw Fast`，在倒數期間切回 Gartic 視窗。
6. 需要停止時按 `Esc` 或程式裡的 `STOP`。

## 建議設定

### 穩定色盤上色

- Mode: `Palette Color`
- Brush Key: `1`
- CPS: `300 ~ 500`
- Line Move ms: `5 ~ 8`
- Image Scale: `80 ~ 90`
- Skip White: 開

如果出現斷線或沒有換色，先把 `Line Move ms` 拉高、CPS 降低。

### RGB 還原度較高

- Mode: `Custom RGB`
- Custom Colors: `16 ~ 32`
- Brush Key: `1`
- CPS: `120 ~ 250`
- Line Move ms: `6 ~ 10`

使用 RGB 模式前，請先打開 Gartic 的自訂色面板，再重新 `Auto Detect`。

### 線稿

- Mode: `Smart Line Art`
- Brush Key: `1 ~ 2`
- Line Detail: `3 ~ 5`
- Stroke Step: `1`

## 常見問題

### 顏色沒有切換

- 固定色盤模式：重新 `Auto Detect`，確認色盤沒有被視窗遮住。
- RGB 模式：確認自訂色面板已打開，且 RGB 輸入框位置沒有被遮擋。
- 降低 CPS 或提高 `Line Move ms`，瀏覽器太忙時可能會漏掉滑鼠/鍵盤事件。

### 畫出來和預覽不一樣

- 繪製期間不要移動瀏覽器視窗或縮放頁面。
- Gartic / 瀏覽器可能漏掉太快的拖曳事件，先用較穩的速度測試。
- `Skip White` 會保留白色畫布，因此原圖的白色高光可能不會被畫出。

### 想更快

- 提高 CPS。
- 降低 `Line Move ms`。
- 降低 `Image Scale`。
- 使用較大的 Brush Key，但細節會變少。

## 專案檔案

- `gartic_auto_drawer.py`：主程式
- `requirements.txt`：Python 依賴套件
- `.gitignore`：忽略快取、虛擬環境與本機測試輸出
