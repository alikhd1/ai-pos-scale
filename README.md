# AI POS Scale (ترازوی هوشمند فروشگاهی)

Desktop point-of-sale application for a **scale + AI camera** terminal (Realtek RTKCam
based "AI camera" module, RS-232 weighing indicator, ESC/POS thermal printer).
Products are recognised by the camera, weighed by the scale and added to an invoice
that is printed on the receipt printer. New products are *trained* at the till in a
few seconds; no data-science workflow is needed.

```
+----------------------------------------------+-------------------------------------+
|  AI Camera (live, tray area outlined)        |  Sales Invoice                      |
|  [ detected item badge / "tray is empty" ]   |  weight 1.234 kg   (scale status)   |
|                                              |  [ Add to Invoice ] [ manual pick ] |
|  Train New Item | Add Angle Sample | Manage  |  item | kg | price/kg | total | x   |
|  Tray Area      | Capture Empty Tray         |  Grand total  ...  Checkout & Print |
+----------------------------------------------+-------------------------------------+
```

## Modules

| file | role |
|------|------|
| `app.py` | PyQt6 GUI, POS logic, tray calibration / training / settings dialogs, `--selftest` |
| `rtk_camera.py` | ctypes binding of `RTKCamSDK.dll` (+ OpenCV fallback, camera thread) |
| `scale_driver.py` | serial scale reader/parser, simulator, live monitor, auto-detect diagnosis |
| `printer_driver.py` | ESC/POS builder, Windows-spooler / COM transports, bitmap receipt renderer (Persian ready) |
| `ai_engine.py` | tray region crop, MobileNetV2 embeddings, `items_db.pkl`, empty-tray reference, matcher |
| `settings_manager.py` | `config.json` load/save with defaults |
| `translations.py` | English / Persian UI strings |
| `build.bat` | one-click PyInstaller build |

## Run from source

```bat
python -m pip install -r requirements.txt
python app.py                 (Persian UI by default; --lang en for English)
python app.py --selftest      (head-less diagnostics -> selftest_report.txt)
python app.py --video clip.avi (use a video file as a virtual camera)
```

## Build the executable

```bat
build.bat            -> dist\AIPosScale\AIPosScale.exe   (recommended, fast start-up)
build.bat /onefile   -> dist\AIPosScale.exe
```

The build embeds `RTKCamSDK.dll`, the MobileNetV2 weights (`models/`) and all Python
dependencies. Copy the `dist\AIPosScale` folder to the POS machine; `config.json`,
`items_db.pkl`, `aipos.log` and the `invoices\` folder are created next to the exe.

## First-time setup at the till

1. **Tray Area** (F4): draw the tray on the live image, either a rectangle (drag) or a
   polygon (click the corners). Everything outside is greyed out, so the surroundings never
   influence recognition. Changing the area invalidates old samples: re-train items and
   capture the empty tray again.
2. **Capture Empty Tray** (F6): with nothing on the tray, press the button. The app stores
   the reference and from then on shows *Tray is empty* instead of guessing a product.
   Repeat after the lighting or the tray changes. When a hardware scale is connected, a
   weight of zero also counts as "empty".
3. **Scale** (Settings, F10): choose the COM port and press *Diagnose (auto-detect)*. It
   listens with the current settings, then tries the other baud rates, parities and the
   common request commands and proposes working settings (*Apply suggested settings*).
   *Start monitor* shows every frame the scale sends (text + hex) together with the parsed
   weight. Use *Copy report* to send the output for support.
4. **Printer** (Settings): Windows printer or COM port, paper width, header/footer,
   *Print Test Receipt*.

## Daily use

1. **Train**: put the product on the tray and press *Train New Item* (F2). Type name and
   price per kg, then press *Capture Sample* (or Space) once per view: turn the product,
   capture again. Three or more views are recommended. Delete a bad thumbnail if needed and
   press *Save Item*. *Add Angle Sample* (F3) adds views to an existing item. Capturing is
   refused while the tray looks empty (unless confirmed).
2. **Sell**: place the product, wait for the green badge and a stable weight, press
   *Add to Invoice* (Enter / Space). Unknown products can be picked manually.
3. **Checkout & Print** (F5): saves `invoices\YYYY-MM\invoice_000123.json` and prints the receipt.

### Scale protocol
The parser understands the common continuous ASCII formats (`ST,GS,+  1.234kg`,
`+001.234 kg`, `   2.500 kg`, STX-framed frames, terminator-less frames separated by an
idle gap, `g`/`lb` units). For request/response indicators set a *poll command*
(e.g. `W\r\n` or `\x05`). For protocols that send raw digits (`001234` = 1.234 kg) set
*implied decimals* = 3. DTR/RTS are asserted on open because many indicators only
transmit when the PC side is "ready".

### Receipt printing
`image` mode renders the receipt with Pillow (Tahoma / Vazirmatn) and sends it as an
ESC/POS raster, so Persian text prints correctly on any ESC/POS printer. `text` mode
sends plain ESC/POS text for Latin receipts. Transports: any Windows printer (RAW spool)
or a serial COM printer.

## Troubleshooting

* **Scale connected but no weight**: Settings → Scale → *Diagnose*. The status pill on the
  main screen distinguishes *no data* (wrong baud / cable / needs poll command) from *data
  not understood* (unknown protocol; send the monitor output).
* **Windows restarts by itself**: a user-mode program cannot restart Windows. Run
  `AIPosScale.exe --selftest`; the *system* section of `selftest_report.txt` lists the
  recent shutdown events: ID 41 / 6008 = power loss, hard reset or overheating,
  ID 1001 = blue screen (driver, typically the camera or USB-serial driver),
  ID 1074 = a program requested the restart (Windows Update). `crash_history.txt` records
  every session that did not end cleanly. To reduce CPU load and heat the app only runs
  the network when the tray image changes (Settings → *Pause recognition while static*).
* **Camera shows OPENCV, not RTK**: expected, see below.

## Notes on PyTorch packaging

PyInstaller does not bundle torchvision's C++ extension (`torchvision._C`) reliably, so
`ai_engine.py` defines MobileNetV2 in pure PyTorch with torchvision-compatible parameter
names and loads the official `mobilenet_v2-b0353104.pth` checkpoint (`strict=True`).
The embeddings are identical to `torchvision.models.mobilenet_v2`; torchvision itself is
not needed at runtime.

## Notes on the Realtek SDK

* `RTKCamSDK.dll` exports `RTKCam_*` functions; the `T_fRTKCam_*` identifiers are the
  function-pointer typedefs Realtek's sample uses after `GetProcAddress`. The wrapper
  exposes both names.
* The DLL is **32-bit** and wraps DirectShow. PyTorch only exists for 64-bit Python, so in
  the 64-bit build the app automatically uses the OpenCV (DirectShow / Media Foundation)
  path to the same UVC camera. The ctypes binding is complete and is used when the
  application runs under a 32-bit interpreter. The SDK contains no recognition features,
  so nothing is lost.
* The prototypes were reverse-engineered from the DLL (no header is shipped); see the
  docstring of `rtk_camera.py`.

## Data files

* `config.json` – all settings (editable by hand), including the tray polygon.
* `items_db.pkl` – trained items (name, price/kg, embeddings, thumbnail) and the empty-tray reference.
* `invoices/` – one JSON per invoice + `counter.json`.
* `aipos.log`, `crash_history.txt`, `selftest_report.txt` – diagnostics.
