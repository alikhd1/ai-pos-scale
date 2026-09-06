# AI POS Scale (ترازوی هوشمند فروشگاهی)

Desktop point-of-sale application for a **scale + AI camera** terminal (Realtek RTKCam
based "AI camera" module, RS-232 weighing indicator, ESC/POS thermal printer).
The camera finds and recognises **up to five products at once**, the scale weighs them,
and the invoice is printed on the receipt printer. New products are *trained* at the
till in a few seconds; no data-science workflow, no labelled dataset, no internet.

```
+----------------------------------------------+-------------------------------------+
|  AI Camera - a labelled box per object        |  Sales Invoice                      |
|  [apple 92%]        [banana 88%]              |  weight 1.234 kg   (scale status)   |
|                                               |  [ Add to Invoice ] [ manual pick ] |
|  Train New Item | Add Angle Sample | Manage   |  item | kg | price/kg | total | x   |
|  Tray Area      | Capture Empty Tray          |  Grand total  ...  Checkout & Print |
+----------------------------------------------+-------------------------------------+
```

## Modules

| file | role |
|------|------|
| `app.py` | PyQt6 GUI, POS logic, tray calibration / training / settings dialogs, `--selftest` |
| `tray_segment.py` | finds up to 5 objects on the tray against the stored empty-tray image |
| `ai_engine.py` | MobileNetV2 embeddings, two-headed matcher, `items_db.pkl`, recognition thread |
| `rtk_camera.py` | ctypes binding of `RTKCamSDK.dll` (+ OpenCV fallback, camera thread) |
| `scale_driver.py` | serial scale reader/parser, simulator, live monitor, auto-detect diagnosis |
| `printer_driver.py` | ESC/POS builder, Windows-spooler / COM transports, Persian bitmap receipts |
| `settings_manager.py` | `config.json` load/save with defaults |
| `translations.py` | English / Persian UI strings |
| `tests/` | head-less regression suites (`python tests/run_all.py`) |
| `build.bat` | one-click PyInstaller build |

## Run from source

```bat
python -m pip install -r requirements.txt
python app.py                 (Persian UI by default; --lang en for English)
python app.py --selftest      (head-less diagnostics -> selftest_report.txt)
python tests/run_all.py       (regression tests, no hardware needed)
```

## Build the executable

```bat
build.bat            -> dist\AIPosScale\AIPosScale.exe   (recommended, fast start-up)
build.bat /onefile   -> dist\AIPosScale.exe
```

Copy the `dist\AIPosScale` folder to the POS machine. `config.json`, `items_db.pkl`,
`aipos.log` and `invoices\` are created next to the exe on first run.

## First-time setup at the till

Do these three steps in order. Skipping step 2 leaves multi-object detection switched off.

1. **Tray Area** (F4). Draw the tray on the live image: drag a rectangle, or click the
   corners for a polygon. Everything outside is greyed out, so the surroundings never
   influence recognition. Changing the area invalidates the empty-tray reference and any
   items trained before it.
2. **Capture Empty Tray** (F6). With nothing on the tray, press it. The app stores nine
   frames as a background model (about 60 KB) plus a per-pixel noise map. This is what
   makes object detection possible, and it is also how the app knows the tray is empty.
   Repeat it after the lighting, the tray or the camera position changes.
3. **Scale** (Settings, F10). Pick the COM port and press *Diagnose (auto-detect)*: it
   listens with the current settings, then tries the other baud rates, parities and the
   common request commands, and proposes working settings. A successful diagnosis also
   switches simulation off for you. *Start monitor* shows every frame the scale sends,
   as text and hex, with the parsed weight beside it.
4. **Printer** (Settings). Windows printer or COM port, paper width, header/footer,
   *Print Test Receipt*.

## Daily use

1. **Train** (F2). Put one product on the tray. Type the name and price per kg, then
   press *Capture Sample* (or Space) once per view: turn the product, capture again.
   Six views are recommended, three is the minimum. Capture count is the single biggest
   accuracy lever measured, far more than anything else. The dialog refuses to capture
   while a hand or a second product is in the frame, because that teaches the wrong thing.
   *Add Angle Sample* (F3) adds views to an item that is already trained.
2. **Sell**. Place the products. Each detected object gets its own box and label. Press
   *Add to Invoice* (Enter / Space). Tap an object in the image first to pick which one.
3. **Checkout & Print** (F5). Saves `invoices\YYYY-MM\invoice_000123.json` and prints.

### What happens with several products on the tray

The scale reports **one** number for everything on it. Splitting that number between
different products by pixel area was measured to be more than 20% wrong on half of all
lines, which is an argument with a customer. So the app never does it:

| on the tray | behaviour |
|---|---|
| one product | priced normally |
| several of the **same** product | one line, weight x price/kg, receipt shows `apple x3` |
| several **different** products | pricing refused, with "weigh them separately" |
| a hand over the tray | pricing refused until it is removed |
| lighting changed too much | pricing refused, prompts to re-capture the empty tray |

Settings > General > *When several products are on the tray* switches this between
`interlock` (default), `quantity` (adds the counting behaviour) and `off` (the old
behaviour, priced regardless).

**Do not trust the object count when items are piled or touching.** Two items overlapping
about a third of their diameter have no visible neck to cut, so the count collapses. This
is why pricing never depends on it.

### Scale protocol
The parser understands the common continuous ASCII formats (`ST,GS,+  1.234kg`,
`+001.234 kg`, `   2.500 kg`, STX-framed frames, terminator-less frames separated by an
idle gap, `g`/`lb` units). For request/response indicators set a *poll command*
(e.g. `W\r\n` or `\x05`). For protocols that send raw digits (`001234` = 1.234 kg) set
*implied decimals* = 3. DTR and RTS are asserted on open because many indicators only
transmit when the PC side is ready. An unstable (moving) weight is always displayed;
"Accept stable readings only" controls whether it may be *invoiced*, not whether it is shown.

### Receipt printing
`image` mode renders the receipt with Pillow and sends it as an ESC/POS raster, so Persian
text prints correctly on any ESC/POS printer. `text` mode sends plain ESC/POS text for
Latin receipts. Transports: any Windows printer (RAW spool) or a serial COM printer.

## Troubleshooting

* **The scale is connected and the test says OK, but no weight appears.** Look at the
  status pill under the weight. `Simulated (COM3: simulation is ON...)` in amber means the
  app is on the simulator: press *Use the real scale COM3*. `Connected, no data` means the
  port is open but silent, so check the baud rate, the cable and whether the indicator needs
  a poll command. `Connected, data not understood` means bytes arrive that the parser cannot
  read: open Settings > Scale > *Start monitor*, press *Copy report* and send it on.
* **No boxes around the objects.** The empty-tray reference is missing (the camera view says
  so). Capture it with F6. It is also cleared automatically whenever the tray area changes.
* **A product is recognised as another one.** Manage Items warns which pairs look alike to
  the camera. Add more views (F3) of both, from the angles that get confused.
* **Everything reads as unknown after a lamp changed.** Re-capture the empty tray, and add a
  few views under the new light. Picking the item by hand also teaches it: each manual pick
  stores that view, capped at five per item, and Manage Items can forget them again.
* **Windows restarts by itself.** A user-mode program cannot restart Windows. Run
  `AIPosScale.exe --selftest`; the *system* section of `selftest_report.txt` lists the recent
  shutdown events: ID 41 or 6008 = power loss, hard reset or overheating, ID 1001 = blue
  screen (a driver, typically the camera or the USB-serial adapter), ID 1074 = a program
  asked for the restart (Windows Update). `crash_history.txt` records every session that did
  not end cleanly.
* **The camera shows OPENCV, not RTK.** Expected, see below.

## How recognition works, and why

Each object is cropped tight against the empty-tray background, masked, and turned into a
1280-d MobileNetV2 embedding. Matching is a two-headed rule: rank the candidates in a
PCA-whitened space, then accept or reject the winner on its top-principal-component-removed
score against a threshold calibrated for that specific item. Measured against the previous
"mean of the top three sample cosines, one global 0.65 threshold":

| change | effect |
|---|---|
| class centroid instead of top-k | hard-set top-1 0.508 -> 0.628, and ten times cheaper |
| remove the top principal component | 0.628 -> 0.674 (it encodes the tray and the light) |
| rank in PCA-whitened space | 0.674 -> 0.756 |
| per-item calibrated thresholds | false accepts on unenrolled produce 1.000 -> 0.078 |

The last row is the important one: with a single global threshold the "unknown item"
verdict did not exist, because raw cosines between crops that share a tray all sit between
0.76 and 0.94. Every one of those numbers is reproduced by `tests/test_matcher.py` on
synthetic data, not quoted from a paper.

## Notes on PyTorch packaging

PyInstaller does not bundle torchvision's C++ extension (`torchvision._C`) reliably, so
`ai_engine.py` defines MobileNetV2 in pure PyTorch with torchvision-compatible parameter
names and loads the official `mobilenet_v2-b0353104.pth` checkpoint with `strict=True`.
The embeddings are identical to `torchvision.models.mobilenet_v2`; torchvision is not
needed at runtime. MobileNetV3-Large was ported and verified as bit-identical too, and was
*not* adopted: it is 13% faster but its 960-d embeddings would force every shop to re-train
every item, and the matcher changes above buy far more accuracy for no migration at all.

## Notes on the Realtek SDK

* `RTKCamSDK.dll` exports `RTKCam_*` functions; the `T_fRTKCam_*` identifiers are the
  function-pointer typedefs Realtek's sample uses after `GetProcAddress`. The wrapper
  exposes both names.
* The DLL is **32-bit** and only wraps DirectShow. PyTorch exists only for 64-bit Python, so
  the 64-bit build reaches the same UVC camera through OpenCV instead. Nothing is lost: the
  SDK contains no recognition features at all. The ctypes binding is complete and is used
  when the application runs under a 32-bit interpreter.
* The prototypes were reverse-engineered from the DLL (no header ships with it); see the
  docstring of `rtk_camera.py`.

## Data files

* `config.json` - all settings, including the tray polygon.
* `items_db.pkl` - version 3: items (name, price, embeddings, capture groups, calibrated
  threshold, crop mode, learned samples) plus the empty-tray reference and image model.
  A version 2 file loads unchanged; its items keep working in whole-tray mode.
* `invoices/` - one JSON per invoice + `counter.json`.
* `aipos.log`, `crash_history.txt`, `selftest_report.txt` - diagnostics.
