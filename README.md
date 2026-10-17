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
| `hid_scale.py` | USB HID scales (the kind that is not a COM port at all) |
| `diagnostics.py` | one-click zip with the frame, masks, crops and scores, for support |
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
3. **Scale** (Settings, F10). Press **Find my scale**. It searches USB HID scales *and*
   every COM port at every common baud rate, parity and request command, then applies
   whatever answered. Use it first; the other buttons are for when it finds nothing.
   *Start monitor* shows every frame the scale sends, as text and hex, with the parsed
   weight beside it, and *Copy report* puts the whole log on the clipboard to send on.

   Not every scale is a serial port. Many integrated POS scales are **USB HID** devices,
   which a serial-only driver cannot see at any baud rate. Those are now supported and
   are what *Automatic* tries first.
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

### Weighing, and catching a tray that does not match the invoice

By default the products **stay on the tray** and each new line is the difference:

| step | tray reads | invoice holds | new line |
|---|---|---|---|
| put product A on | 0.100 | 0.000 | 0.100 |
| add A to the invoice | 0.100 | 0.100 | - |
| put product B on as well | 0.300 | 0.100 | 0.200 |
| add B to the invoice | 0.300 | 0.300 | - |

The line under the weight always shows *on the tray* against *on the invoice*. Because
everything billed should still be on the pan and nothing else should be, any disagreement
is visible immediately and is checked again at checkout: an unbilled item slipped on, or
one swapped for a heavier one after being scanned, shows up as a weight that does not add
up, and finishing asks for confirmation. Settings > General sets the allowed difference
(20 g by default) and whether a mismatch blocks checkout.

Switch to *Tray is emptied between items* if your staff prefer to weigh one product at a
time; then each line is simply the whole tray and no reconciliation is done.

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

## Remote support, for a till you cannot walk up to

A POS terminal lives in a shop, behind a router nobody controls, and the person who has
to fix it is usually somewhere else. Switched on, the app posts a snapshot of itself to a
service you run, every ten minutes by default: whether the camera is working and which
picture formats it tried, what the scale reads, the exact banner on the screen and why
the till is refusing to sell, the invoice, the catalogue size, the tail of the log and a
small picture of the camera view.

The address and the device key are baked into the build, in
[`support_config.py`](support_config.py), so a till installed in a shop starts reporting
with nothing typed in. **Settings > Remote support** shows which address is in use, lists
exactly what leaves the machine with each part switchable on its own, and lets a
particular till be pointed elsewhere or switched off without a new build. Blank out
`support_config.py` and a build sends nothing until somebody fills the settings in.

A token inside a shipped executable is not really a secret - anyone with the exe can read
it out. That is why the service has two keys: the one in the build only lets a till post
its own status, and cannot read another till, open the dashboard, or queue an action. The
admin token that can do those things never leaves the server.

TLS is verified against a CA bundle shipped with the app (`certifi`), not the machine's
certificate store. This is not belt-and-braces: Python on Windows reads only the roots
already cached in the Windows store and, unlike a browser, never triggers Windows' fetch
-this-root-on-demand update, so a till that has never pulled down the relevant root fails
to verify a perfectly good certificate while the same URL opens fine in Edge on that same
machine. Point `support.ca_file` at your own bundle if you terminate TLS yourself.

The service can also queue a short list of support actions, which the till picks up at
its next check-in: `ping, snapshot, selftest, find_camera, find_scale, restart_camera,
restart_scale, diagnostics, logs, get_setting, set_setting, screenshot`. That list is
closed. **This is not a remote shell and must not become one** - there is no "run this
command", no download-and-execute, and `set_setting` only accepts a key that already
exists. A till handles money; a general-purpose backdoor on one is a liability for
whoever installs it, and this fixed list buys nearly all of the usefulness for a small
fraction of the risk. Connections are outbound HTTPS only (plain HTTP is refused unless
it is the loopback), nothing listens on a port, and `allow_actions` turns even the listed
ones off, leaving a report-only channel.

The other end is in [`support_server/`](support_server/): one stdlib-only Python file, no
pip, no virtualenv. `setup-ubuntu.sh` installs it under systemd on the loopback and adds
one `location` block to an nginx site you already have a certificate for, so no new DNS
record and no new certificate are needed:

```bash
sudo ./setup-ubuntu.sh --site /etc/nginx/sites-available/your-site.conf --path /pos
```

It generates two separate tokens - the one the tills carry, and the admin one that the
dashboard and the action queue need - and prints them once. The dashboard is a single
page showing every till, its camera picture, its state, and a button per action.

## Troubleshooting

* **The till says the certificate could not be verified.** Check the clock first: a date
  more than a day out fails every certificate. The app trusts its own bundle rather than
  the Windows store, so a stale store is no longer a cause, but a corporate proxy that
  re-signs TLS is - point `support.ca_file` at that proxy's root.
* **A till stopped reporting to the support service.** The dashboard greys out tills that
  have not checked in. Nothing can be done from this end: the channel is outbound only, so
  if the shop's internet is down or the app is closed, somebody has to be there. The till
  keeps the results of any action it ran and sends them when it gets back.
* **The screen says "No camera".** The till stays usable: it drops into manual mode, the
  banner says so, and you can still pick a product from the list and weigh it. Press
  *Try the camera again* next to the pill - it tries every device index on the machine,
  and every picture format, and saves the one that works.
* **The camera "opens but no frames".** That is a camera that answers Windows but will not
  deliver the picture format it was asked for, usually because MJPEG was forced on a camera
  that only does YUY2. The app now tries the device's own format first and only then the
  alternatives, so this should be gone; if it comes back, `AIPosScale.exe --selftest` lists
  every combination it tried, and which device indexes exist.
* **"cameras Windows can see: NONE" in the self-test.** Then the problem is below the app:
  the camera is unplugged, has no driver, or is not a standard USB camera at all. Check
  Device Manager first - nothing in the app can fix a camera Windows cannot see.
* **The scale says "Simulated" and there is no COM port set.** Press *Find my scale* under
  the weight. It tries every USB HID scale and every COM port at the usual baud rates, and
  applies whatever answers.
* **The scale is connected and the test says OK, but no weight appears.** Look at the
  status pill under the weight. `Simulated (COM3: simulation is ON...)` in amber means the
  app is on the simulator: press *Use the real scale COM3*. `Connected, no data` means the
  port is open but silent, so check the baud rate, the cable and whether the indicator needs
  a poll command. `Connected, data not understood` means bytes arrive that the parser cannot
  read: open Settings > Scale > *Start monitor*, press *Copy report* and send it on.
* **No boxes around the objects.** The empty-tray reference is missing (the camera view says
  so). Capture it with F6. It is also cleared automatically whenever the tray area changes.
* **A product is recognised as another one.** Manage Items warns which pairs look alike to
  the camera. Add more views (F3) of both, from the angles that get confused. When the app
  is unsure it now lists the alternatives under the picture rather than guessing; tapping
  the right one both bills it and teaches the model that view.
* **It does not detect my goods properly.** Press **Save diagnostics**. It writes a zip
  next to the program containing the camera frame, the tray outline, the exact crop each
  object is recognised from, the masks, every candidate score, and the catalogue. Send
  that file: segmentation and recognition faults depend entirely on what your camera sees
  on your counter, and this is what makes them fixable.
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

**Two thresholds, not one.** A single cut-off has to choose between rejecting correct
products and billing wrong ones, and either choice is bad at a till. So each item gets
both an *accept* level and a lower *offer* level, each calibrated from that item's own
data: how well a held-out view of it scores, and how well every other item scores against
it.

| score | what happens |
|---|---|
| above the accept level | billed straight away, no tap |
| between the two | the item and its look-alikes are listed under the picture: "Is it X?" |
| below the offer level | unknown item |

On the test set that gives 83% of queries billed with no tap, the correct product offered
first for 94%, and **nothing that was never enrolled is ever billed automatically**. When
two of your products genuinely look alike to the camera, you get a choice instead of a
silent guess, and the one you tap is also remembered as a new view of that product.

## Speed

The model is traced and optimized for the machine it runs on, which measured **about
1.7x faster** per object in the shipped build - 40 ms down to 24 ms for a whole
recognition, repeatable over three runs on a settled machine - with numerically
identical output, so nothing about accuracy changes. Expect a lower figure from a
single run taken right after a build, while the machine is still busy. It happens on a background thread, so the
till works immediately and switches over when it is ready. Turn it off with
`ai.fast_inference` if a machine ever misbehaves.

Two things measured and *rejected*, so nobody repeats them: batching several crops into
one forward pass is not faster on this CPU, and shrinking the network input below 224
costs more accuracy than it buys speed. Preprocessing is only about 2 ms of a call, so
there is nothing to win there either - the time is in the network.

Beyond that, the network is simply not run when it does not need to be: not while the
tray image is unchanged, not for an object whose box has not moved, and never more than
three new objects in one frame. Segmentation itself costs about 20 ms.

Check it on your own machine: `AIPosScale.exe --selftest` prints a line like
`speed per object: 38 ms -> 25 ms after optimizing for this CPU (1.5x, built in 3s)`.
If it says the model could *not* be optimized, recognition is running slower than it
should and `aipos.log` says why. The optimizer needs `sympy`, which must be in the build
(`torch.jit.trace` imports it) - excluding it disables the speed-up silently, because the
app correctly falls back to the slower path instead of failing.

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
* `aipos_diagnostics_*.zip` - written by *Save diagnostics*; safe to delete, and safe
  to send on (images of the tray and the settings, no personal or payment data).
