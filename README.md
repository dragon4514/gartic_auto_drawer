# Gartic OpenCV Drawer

Gartic OpenCV Drawer 是一個用 OpenCV、PyAutoGUI 和 PySide6 製作的 Gartic Phone 自動畫圖工具。它可以把圖片轉成線稿、固定色盤色塊、自訂 RGB 色塊或筆觸渲染，並透過滑鼠與鍵盤事件在 Gartic Phone 畫布上繪製。

> 這是個人輔助工具。使用時請尊重遊戲規則、房間規範與其他玩家體驗。

## 安裝與啟動

本專案目前只提供原始碼版本。請先安裝 Python 3.10 以上，再用 PowerShell 執行：

```powershell
git clone https://github.com/dragon4514/gartic_auto_drawer.git
cd gartic_auto_drawer
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python gartic_auto_drawer.py
```

## 重要提醒

強烈建議在瀏覽器安裝廣告阻擋套件後再使用。Gartic Phone 頁面上的廣告可能造成瀏覽器卡頓，讓滑鼠拖曳、快捷鍵或換色事件被漏掉，最後導致畫出來和預覽不同。

繪製期間不要移動瀏覽器視窗、縮放頁面或切換畫布位置。程式是依照偵測到的螢幕座標操作，畫面位置改變後座標就會失準。

## 功能

- 自動偵測 Gartic Phone 畫布、固定 18 色色盤、筆刷按鈕與 RGB 面板位置
- 圖片預覽，開始前先確認線稿、色塊與實際畫布比例
- 支援 Smart Line Art、固定色盤、Custom RGB、SBR 筆觸渲染等模式
- 支援 `Esc` 與程式內 `STOP` 按鈕緊急停止
- 支援 Gartic 筆刷快捷鍵，`Brush Key 0` 會使用反引號快捷鍵選擇更細的線
- 可調整 CPS、筆刷大小、圖片縮放、線條細節、移動速度、RGB 顏色數與 RGB 面板延遲
- 可保存畫布、色盤、筆刷與 RGB 面板校正設定
- 可用 Overlay 手動拖曳校正偵測結果

## 系統需求

- Windows 10 / 11
- 瀏覽器可正常開啟 Gartic Phone
- Python 3.10 以上
- 第一次安裝依賴時需要網路下載 Python 套件

## 基本使用流程

1. 開啟 Gartic Phone 繪圖頁面，確認畫布、色盤、筆刷列都看得到。
2. 建議先安裝廣告阻擋套件，並關掉會讓頁面卡頓的分頁或程式。
3. 執行 Gartic OpenCV Drawer。
4. 點 `Load Image` 載入圖片。
5. 點 `Auto Detect` 自動偵測畫布、色盤、筆刷與 RGB 面板。
6. 如果偵測位置不準，使用 Overlay 手動拖曳校正。
7. 選擇模式並點 `Preview` 檢查效果。
8. 點 `Draw Fast`，讓 Gartic 視窗保持在前景。
9. 需要停止時按 `Esc` 或程式裡的 `STOP`。

## 繪圖模式

| 模式 | 用途 | 說明 |
| --- | --- | --- |
| `Smart Line Art` | 整合線稿 | 自動整理圖片輪廓與主要細節，適合想先保留線條的圖片 |
| `Palette Color` | 固定色盤全彩 | 使用 Gartic 18 色色盤，穩定、速度快 |
| `Custom RGB` | 自訂 RGB 顏色 | 顏色還原度較高，但換色較慢、較吃瀏覽器穩定度 |
| `SBR` | 筆觸渲染 | 實驗性模式，適合做粗略筆觸效果 |

## 建議設定

### 穩定固定色盤上色

- Mode: `Palette Color`
- Brush Key: `1`
- CPS: `300 ~ 500`
- Line Move ms: `5 ~ 8`
- Image Scale: `80 ~ 90`
- Stroke Step: `1`
- Skip White: 開

如果出現斷線、漏色或沒有換色，先把 `Line Move ms` 拉高，或把 CPS 降低。

### RGB 還原度較高

- Mode: `Custom RGB`
- Custom Colors: `16 ~ 64`
- Brush Key: `1`
- CPS: `120 ~ 250`
- Line Move ms: `6 ~ 10`
- RGB Panel ms: `300` 以上

Custom RGB 最高支援到 384 色，但顏色越多，換色次數越多，也越容易受瀏覽器卡頓影響。一般圖片建議先用 24、48 或 64 色測試。

使用 RGB 模式前，請先打開 Gartic 的自訂色面板，再重新 `Auto Detect`。如果別人的電腦解析度或瀏覽器縮放不同，請用 Overlay 手動校正 RGB 色塊與 R/G/B 輸入框。

### 線稿

- Mode: `Smart Line Art`
- Brush Key: `0 ~ 2`
- Line Detail: `3 ~ 5`
- Line Move ms: `5 ~ 12`
- Stroke Step: `1`

`Brush Key 0` 是額外細線，會先讓 Gartic 視窗取得焦點，再按反引號快捷鍵。若快捷鍵沒有生效，請確認 Gartic 視窗在繪製時保持前景。

## 設定檔

校正資料會存在 `profiles/gartic_profiles.json`。

`profiles/` 會建立在專案資料夾內。這樣別人下載後可以在自己的專案資料夾保存自己的螢幕座標，不會套用到你的本機路徑。

`profiles/` 已加入 `.gitignore`，不會提交到 GitHub。

## 常見問題

### Auto Detect 找不到畫布

- 確認 Gartic Phone 繪圖頁面在螢幕上可見。
- 不要讓其他視窗遮住畫布、色盤或筆刷列。
- 瀏覽器縮放建議先用 100%。
- 如果偵測到的位置不準，使用 Overlay 手動校正並保存 profile。

### 顏色沒有切換

- 固定色盤模式：重新 `Auto Detect`，確認色盤沒有被遮住。
- RGB 模式：確認自訂色面板已打開，且 R/G/B 輸入框位置正確。
- 提高 `RGB Panel ms` 和 `Line Move ms`。
- 降低 CPS，瀏覽器太忙時可能漏掉點擊、拖曳或鍵盤事件。

### 畫出來和預覽不一樣

- 繪製期間不要移動瀏覽器視窗或縮放頁面。
- 安裝廣告阻擋套件，避免廣告造成頁面卡頓。
- 預覽是根據筆刷大小估算的結果，Gartic 實際筆刷、瀏覽器縮放與拖曳事件採樣都可能造成差異。
- `Skip White` 會保留白色畫布，因此原圖的白色高光、文字或空洞可能不會被畫出。

### 線條或色塊斷掉

- 把 `Line Move ms` 調高。
- 把 CPS 降低。
- 使用較大的 Brush Key。
- 避免在 CPU 很忙或瀏覽器很卡的時候繪製。

### 想更快

- 提高 CPS，但 300 以上不保證每台電腦都穩。
- 降低 `Line Move ms`。
- 降低 `Image Scale`。
- 使用較大的 Brush Key。
- 減少 Custom RGB 顏色數。

## 專案結構

```text
gartic_auto_drawer.py          完整單檔主程式，內部用區塊整理設定、偵測、影像處理、UI 與自動繪製流程
requirements.txt               Python 依賴套件
```

## 開發檢查

修改程式後可先執行：

```powershell
python -m py_compile gartic_auto_drawer.py
python -c "import gartic_auto_drawer; print('import ok')"
```

這只能檢查語法與基本匯入，實際畫布偵測與繪圖仍建議在 Gartic Phone 測試。

## 注意事項

- 本工具依賴螢幕座標與瀏覽器事件，不是 Gartic Phone 官方 API。
- 不同解析度、瀏覽器縮放、Windows 顯示比例都可能需要重新校正。
- 高 CPS 不代表 Gartic 一定能接收所有事件，穩定度通常比極限速度重要。
- 請不要在公開房間濫用，避免影響其他玩家。
