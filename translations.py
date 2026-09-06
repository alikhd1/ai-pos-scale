"""
translations.py
---------------
Minimal bilingual (English / Persian) string table for the GUI.

``tr("key")`` returns the text for the active language and falls back to the
English text, then to the key itself, so a missing translation never crashes
the application.
"""
from __future__ import annotations

_LANG = "en"

STRINGS = {
    # ---- window / general ----
    "app_title": ("AI POS Scale", "ترازوی هوشمند فروشگاهی"),
    "camera_panel": ("AI Camera", "دوربین هوشمند"),
    "invoice_panel": ("Sales Invoice", "فاکتور فروش"),
    "settings": ("Settings", "تنظیمات"),
    "ok": ("OK", "تأیید"),
    "cancel": ("Cancel", "انصراف"),
    "close": ("Close", "بستن"),
    "save": ("Save", "ذخیره"),
    "apply": ("Apply", "اعمال"),
    "yes": ("Yes", "بله"),
    "no": ("No", "خیر"),
    "error": ("Error", "خطا"),
    "warning": ("Warning", "هشدار"),
    "info": ("Information", "اطلاعات"),
    "confirm": ("Confirm", "تأیید"),
    "delete": ("Delete", "حذف"),
    "edit": ("Edit", "ویرایش"),
    "name": ("Name", "نام"),
    "price_per_kg": ("Price / Kg", "قیمت هر کیلو"),
    "samples": ("Samples", "نمونه‌ها"),
    "loading_model": ("Loading AI model...", "بارگذاری مدل هوش مصنوعی..."),
    "model_ready": ("AI model ready", "مدل هوش مصنوعی آماده است"),
    "model_failed": ("AI model failed to load", "بارگذاری مدل هوش مصنوعی ناموفق بود"),

    # ---- camera / recognition ----
    "no_camera": ("No camera", "دوربین در دسترس نیست"),
    "camera_backend": ("Camera", "دوربین"),
    "place_item": ("Place an item on the scale", "کالا را روی ترازو قرار دهید"),
    "unknown_item": ("Unknown item", "کالای ناشناس"),
    "confidence": ("Confidence", "اطمینان"),
    "train_new_item": ("Train New Item", "آموزش کالای جدید"),
    "add_angle_sample": ("Add Angle Sample", "افزودن نمونه از زاویه دیگر"),
    "manage_items": ("Manage Items", "مدیریت کالاها"),
    "training": ("Training", "آموزش"),
    "training_progress": ("Capturing sample {done} / {total}", "ثبت نمونه {done} از {total}"),
    "training_hint": ("Slowly rotate the item while samples are captured.",
                      "در حین ثبت نمونه‌ها، کالا را به‌آرامی بچرخانید."),
    "training_done": ("Item '{name}' trained with {count} samples.",
                      "کالای «{name}» با {count} نمونه آموزش داده شد."),
    "training_failed": ("Training failed: {error}", "آموزش ناموفق بود: {error}"),
    "start": ("Start", "شروع"),
    "item_name": ("Item name", "نام کالا"),
    "sample_count": ("Samples to capture", "تعداد نمونه"),
    "select_item": ("Select item", "انتخاب کالا"),
    "no_items_trained": ("No items trained yet. Use 'Train New Item' first.",
                         "هنوز کالایی آموزش داده نشده است. ابتدا «آموزش کالای جدید» را بزنید."),
    "invalid_name": ("Please enter an item name.", "لطفاً نام کالا را وارد کنید."),
    "invalid_price": ("Please enter a valid price.", "لطفاً قیمت معتبر وارد کنید."),
    "delete_item_confirm": ("Delete item '{name}' and all its samples?",
                            "کالای «{name}» و همه نمونه‌های آن حذف شود؟"),
    "items_count": ("{count} items", "{count} کالا"),

    # ---- scale ----
    "weight": ("Weight", "وزن"),
    "kg": ("kg", "کیلوگرم"),
    "scale_status": ("Scale", "ترازو"),
    "scale_simulated": ("Simulated", "شبیه‌سازی"),
    "scale_connected": ("Connected", "متصل"),
    "scale_disconnected": ("Disconnected", "قطع"),
    "scale_stable": ("Stable", "ثابت"),
    "scale_unstable": ("Unstable", "ناپایدار"),
    "simulated_weight": ("Simulated weight (kg)", "وزن شبیه‌سازی‌شده (کیلوگرم)"),
    "zero_weight": ("Weight is zero. Place the item on the scale.",
                    "وزن صفر است. کالا را روی ترازو قرار دهید."),

    # ---- invoice ----
    "add_to_invoice": ("Add to Invoice", "افزودن به فاکتور"),
    "add_manual": ("Add manually", "افزودن دستی"),
    "col_item": ("Item", "کالا"),
    "col_weight": ("Weight (kg)", "وزن (کیلوگرم)"),
    "col_price": ("Price/Kg", "قیمت هر کیلو"),
    "col_total": ("Total", "جمع"),
    "col_action": ("", ""),
    "delete_row": ("Delete Row", "حذف ردیف"),
    "clear_invoice": ("Clear", "پاک کردن"),
    "checkout": ("Checkout & Print", "تسویه و چاپ"),
    "grand_total": ("Grand Total", "جمع کل"),
    "invoice_empty": ("Invoice is empty.", "فاکتور خالی است."),
    "clear_confirm": ("Remove all rows from the invoice?", "همه ردیف‌های فاکتور حذف شوند؟"),
    "checkout_done": ("Invoice #{number} saved. Total: {total}", "فاکتور شماره {number} ذخیره شد. جمع: {total}"),
    "print_failed": ("Printing failed: {error}", "چاپ ناموفق بود: {error}"),
    "printed": ("Receipt sent to printer.", "رسید به چاپگر ارسال شد."),
    "no_item_detected": ("No item recognised. Pick one manually.", "کالایی شناسایی نشد. به‌صورت دستی انتخاب کنید."),
    "invoice_no": ("Invoice", "فاکتور"),
    "date": ("Date", "تاریخ"),
    "items": ("Items", "اقلام"),
    "receipt_kg": ("Kg", "کیلو"),
    "currency_decimals": ("Currency decimals", "اعشار قیمت"),

    # ---- settings dialog ----
    "tab_general": ("General", "عمومی"),
    "tab_camera": ("Camera & AI", "دوربین و هوش مصنوعی"),
    "tab_scale": ("Scale", "ترازو"),
    "tab_printer": ("Printer", "چاپگر"),
    "language": ("Language", "زبان"),
    "store_name": ("Store name", "نام فروشگاه"),
    "currency": ("Currency", "واحد پول"),
    "auto_add": ("Auto-add recognised item when weight is stable", "افزودن خودکار کالای شناسایی‌شده هنگام ثبات وزن"),
    "restart_note": ("Some changes apply after the dialog is closed.", "برخی تغییرات پس از بستن پنجره اعمال می‌شوند."),
    "backend": ("Backend", "درگاه دوربین"),
    "device_index": ("Device index", "شماره دستگاه"),
    "resolution": ("Resolution", "وضوح تصویر"),
    "mirror": ("Mirror image", "آینه کردن تصویر"),
    "rotate": ("Rotate", "چرخش"),
    "roi_size": ("Target area size", "اندازه ناحیه هدف"),
    "threshold": ("Confidence threshold", "حد آستانه اطمینان"),
    "smoothing": ("Smoothing frames", "فریم‌های هموارسازی"),
    "train_samples": ("Samples per training", "نمونه در هر آموزش"),
    "video_file": ("Video file (virtual camera)", "فایل ویدیو (دوربین مجازی)"),
    "browse": ("Browse...", "انتخاب..."),
    "com_port": ("COM port", "پورت COM"),
    "scan_ports": ("Scan", "جستجو"),
    "baudrate": ("Baud rate", "سرعت (Baud)"),
    "data_bits": ("Data bits", "بیت داده"),
    "parity": ("Parity", "توازن"),
    "stop_bits": ("Stop bits", "بیت توقف"),
    "poll_command": ("Poll command (optional)", "فرمان درخواست وزن (اختیاری)"),
    "unit": ("Unit", "واحد"),
    "implied_decimals": ("Implied decimals", "اعشار ضمنی"),
    "stable_only": ("Accept stable readings only", "فقط وزن ثابت پذیرفته شود"),
    "simulate_scale": ("Simulate scale (no hardware)", "شبیه‌سازی ترازو (بدون سخت‌افزار)"),
    "test_scale": ("Test Scale Connection", "تست اتصال ترازو"),
    "scale_test_ok": ("Scale OK: {weight} kg  (raw: {raw})", "ترازو سالم است: {weight} کیلوگرم (خام: {raw})"),
    "scale_test_fail": ("Scale test failed: {error}", "تست ترازو ناموفق: {error}"),
    "printer_interface": ("Interface", "نوع اتصال"),
    "printer_name": ("Windows printer", "چاپگر ویندوز"),
    "paper_width": ("Paper width", "عرض کاغذ"),
    "print_mode": ("Print mode", "حالت چاپ"),
    "header_text": ("Header text", "متن سربرگ"),
    "footer_text": ("Footer text", "متن پابرگ"),
    "cut_paper": ("Cut paper after receipt", "برش کاغذ پس از رسید"),
    "print_test": ("Print Test Receipt", "چاپ رسید آزمایشی"),
    "print_test_ok": ("Test receipt sent.", "رسید آزمایشی ارسال شد."),
    "enable_printer": ("Enable printer", "فعال‌سازی چاپگر"),
    "mode_image": ("Image (any language)", "تصویری (همه زبان‌ها)"),
    "mode_text": ("Text (ESC/POS, Latin only)", "متنی (ESC/POS، فقط لاتین)"),
    "iface_windows": ("Windows printer (spooler)", "چاپگر ویندوز"),
    "iface_serial": ("Serial COM", "سریال COM"),
    "iface_file": ("File (debug)", "فایل (اشکال‌زدایی)"),
    "iface_none": ("None", "هیچ"),
    "backend_auto": ("Auto (RTK SDK, then OpenCV)", "خودکار (RTK SDK سپس OpenCV)"),
    "backend_rtk": ("RTKCam SDK only", "فقط RTKCam SDK"),
    "backend_opencv": ("OpenCV only", "فقط OpenCV"),
    "settings_saved": ("Settings saved.", "تنظیمات ذخیره شد."),

    # ---- tray area / empty tray ----
    "tray_calibrate": ("Tray Area", "ناحیه سینی"),
    "tray_calibrate_title": ("Calibrate tray area", "تعیین ناحیه سینی"),
    "tray_hint_poly": ("Click on the image to add corner points around the tray. Drag a point to move it; right-click removes the last point.",
                       "روی تصویر کلیک کنید تا نقاط دور سینی اضافه شوند. برای جابه‌جایی، نقطه را بکشید؛ کلیک راست آخرین نقطه را حذف می‌کند."),
    "tray_hint_rect": ("Drag a rectangle around the tray.", "با کشیدن ماوس، مستطیلی دور سینی رسم کنید."),
    "mode_polygon": ("Polygon", "چندضلعی"),
    "mode_rect": ("Rectangle", "مستطیل"),
    "mask_outside": ("Grey out the area outside the tray", "بیرون سینی خاکستری شود"),
    "reset_default": ("Reset", "بازنشانی"),
    "model_view": ("What the model sees", "تصویر ورودی مدل"),
    "tray_saved": ("Tray area saved. Items trained before may need re-training. Capture the empty tray again.",
                   "ناحیه سینی ذخیره شد. کالاهای قبلی ممکن است نیاز به آموزش مجدد داشته باشند. سینی خالی را دوباره ثبت کنید."),
    "tray_points_needed": ("Add at least 3 points or draw a rectangle (or Reset for the default square).",
                           "حداقل ۳ نقطه اضافه کنید یا مستطیل بکشید (یا بازنشانی برای مربع پیش‌فرض)."),
    "capture_empty_tray": ("Capture Empty Tray", "ثبت سینی خالی"),
    "empty_tray_confirm": ("Remove everything from the tray, then press OK to capture the empty-tray reference.",
                           "همه چیز را از روی سینی بردارید و سپس تأیید را بزنید تا مرجع سینی خالی ثبت شود."),
    "empty_tray_saved": ("Empty-tray reference saved ({count} samples).", "مرجع سینی خالی ثبت شد ({count} نمونه)."),
    "empty_tray_weight_warning": ("The scale reports weight on the tray. Remove the item first.", "ترازو وزن نشان می‌دهد. ابتدا کالا را بردارید."),
    "tray_empty": ("Tray is empty", "سینی خالی است"),
    "tray_empty_no_ref": ("Empty tray not captured yet", "سینی خالی هنوز ثبت نشده است"),
    "empty_threshold": ("Empty-tray sensitivity", "حساسیت تشخیص سینی خالی"),
    "motion_gate": ("Pause recognition while the image is static (saves CPU)", "توقف تشخیص هنگام ثابت بودن تصویر (کاهش مصرف CPU)"),

    # ---- manual training ----
    "capture_sample": ("Capture Sample", "گرفتن نمونه"),
    "save_item": ("Save Item", "ذخیره کالا"),
    "samples_captured": ("{count} samples captured", "{count} نمونه گرفته شد"),
    "need_samples": ("Capture at least {min} samples from different angles.", "حداقل {min} نمونه از زاویه‌های مختلف بگیرید."),
    "delete_sample": ("Delete selected sample", "حذف نمونه انتخاب‌شده"),
    "auto_capture": ("Auto capture every {ms} ms", "گرفتن خودکار هر {ms} میلی‌ثانیه"),
    "training_blocked_empty": ("The tray looks empty. Capture anyway?", "سینی خالی به نظر می‌رسد. با این حال نمونه گرفته شود؟"),
    "capture_hint": ("Place the item, change its angle, press Capture. Repeat from several angles (Space = capture).",
                     "کالا را بگذارید، زاویه را تغییر دهید و «گرفتن نمونه» را بزنید. از چند زاویه تکرار کنید (کلید Space = گرفتن)."),
    "no_frame": ("No camera frame available.", "تصویری از دوربین در دسترس نیست."),
    "few_samples_confirm": ("Only {count} samples. Save anyway?", "فقط {count} نمونه. با این حال ذخیره شود؟"),

    # ---- scale diagnostics ----
    "serial_monitor": ("Serial monitor", "مانیتور سریال"),
    "start_monitor": ("Start monitor", "شروع مانیتور"),
    "stop_monitor": ("Stop monitor", "توقف مانیتور"),
    "diagnose_scale": ("Diagnose (auto-detect)", "عیب‌یابی خودکار"),
    "apply_suggestion": ("Apply suggested settings", "اعمال تنظیمات پیشنهادی"),
    "scale_no_data": ("Connected, no data", "متصل، بدون داده"),
    "scale_bad_data": ("Connected, data not understood", "متصل، داده ناشناخته"),
    "scale_connecting": ("Connecting", "در حال اتصال"),
    "scale_hold": ("Hold last weight (s)", "نگه‌داشتن آخرین وزن (ثانیه)"),
    "copy_report": ("Copy report", "کپی گزارش"),
    "report_copied": ("Report copied to clipboard.", "گزارش در کلیپ‌بورد کپی شد."),
    "diag_running": ("Diagnosing, please wait...", "در حال عیب‌یابی، لطفاً صبر کنید..."),

    # ---- scale mode / stability ----
    "scale_sim_overrides_port": ("simulation is ON, the port is ignored", "شبیه‌سازی روشن است، پورت نادیده گرفته می‌شود"),
    "scale_stale_data": ("Connected, weight is stale", "متصل، وزن قدیمی است"),
    "use_real_scale": ("Use the real scale {port}", "استفاده از ترازوی واقعی {port}"),
    "weight_unstable": ("Weight is not stable yet", "وزن هنوز ثابت نشده است"),
    "scale_mode": ("Scale mode", "حالت ترازو"),
    "sim_unticked": ("hardware answered, simulation switched off", "سخت‌افزار پاسخ داد، شبیه‌سازی خاموش شد"),

    # ---- multi-object detection ----
    "objects_detected": ("{count} objects", "{count} کالا"),
    "one_object": ("1 object", "۱ کالا"),
    "multi_item_block": ("More than one product on the tray. Weigh them separately.",
                         "بیش از یک کالا روی سینی است. جداگانه وزن کنید."),
    "hand_detected": ("Move your hand away from the tray", "دست خود را از روی سینی بردارید"),
    "lighting_changed": ("Lighting changed. Capture the empty tray again.",
                         "نور تغییر کرده است. سینی خالی را دوباره ثبت کنید."),
    "capture_empty_first": ("Capture the empty tray to enable multi-item detection",
                            "برای تشخیص چند کالا، سینی خالی را ثبت کنید"),
    "same_item_qty": ("{count} x {name}", "{count} عدد {name}"),
    "select_object_hint": ("Tap an object in the image to select it", "برای انتخاب، روی کالا در تصویر بزنید"),
    "per_object_recognition": ("Recognise each object separately (recommended)",
                               "تشخیص جداگانه هر کالا (توصیه می‌شود)"),
    "multi_item_mode": ("When several products are on the tray", "وقتی چند کالا روی سینی است"),
    "mode_off": ("Price anyway (old behaviour)", "با همین وضع فاکتور شود (رفتار قبلی)"),
    "mode_interlock": ("Refuse to price a mixed tray", "فاکتور کردن سینی مخلوط ممنوع"),
    "mode_quantity": ("Refuse mixed, count identical items", "مخلوط ممنوع، کالاهای یکسان شمرده شوند"),
    "max_objects": ("Maximum objects per frame", "بیشترین تعداد کالا در هر فریم"),
    "detection_off": ("Object detection off", "تشخیص کالا خاموش"),

    # ---- training ----
    "capture_more_views": ("{done} of {want} views. More views, better accuracy.",
                           "{done} از {want} نما. هر چه نما بیشتر، دقت بالاتر."),
    "capture_blocked_multi": ("More than one object in the frame. Leave only the product you are training.",
                              "بیش از یک کالا در تصویر است. فقط کالایی که آموزش می‌دهید بماند."),
    "capture_blocked_hand": ("Your hand is in the frame. Move it away and capture again.",
                             "دست شما در تصویر است. آن را کنار ببرید و دوباره بگیرید."),
    "object_mode_note": ("Samples are taken from the detected object only.",
                         "نمونه‌ها فقط از خود کالای شناسایی‌شده گرفته می‌شوند."),
    "tray_mode_note": ("No empty-tray reference: samples are taken from the whole tray.",
                       "مرجع سینی خالی وجود ندارد: نمونه‌ها از کل سینی گرفته می‌شوند."),
    "retrain_needed": ("This item was trained on the whole tray. Re-train it for the new per-object accuracy.",
                       "این کالا با تصویر کل سینی آموزش دیده است. برای دقت جدید، دوباره آموزش دهید."),
    "mode_column": ("Mode", "حالت"),
    "learned": ("Learned", "آموخته"),
    "forget_learned": ("Forget learned samples", "حذف نمونه‌های آموخته"),
    "confusable_items": ("These items look alike to the camera: {pairs}",
                         "این کالاها برای دوربین شبیه هم هستند: {pairs}"),
    "reinforced": ("Learned this view of {name}", "این نمای {name} آموخته شد"),
    "learn_manual": ("Learn from manual item picks", "یادگیری از انتخاب دستی کالا"),

    # ---- camera backend ----
    "rtk_note": ("RTKCamSDK.dll is 32-bit and cannot be loaded inside this 64-bit application; the same camera is used through OpenCV.",
                 "فایل RTKCamSDK.dll سی‌ودو بیتی است و در این برنامه‌ی ۶۴ بیتی بارگذاری نمی‌شود؛ همان دوربین از طریق OpenCV استفاده می‌شود."),
    "rtk_fallback_status": ("RTK SDK unavailable (32-bit DLL); camera via OpenCV", "SDK دوربین در دسترس نیست (DLL سی‌ودو بیتی)؛ دوربین از طریق OpenCV"),
    "unclean_shutdown": ("The previous session ended unexpectedly (Windows restart or crash). Run --selftest for details.",
                         "اجرای قبلی به‌طور غیرمنتظره پایان یافت (ری‌استارت ویندوز یا خطا). برای جزئیات selftest را اجرا کنید."),
}


def set_language(lang: str) -> None:
    global _LANG
    _LANG = "fa" if str(lang).lower().startswith("fa") else "en"


def get_language() -> str:
    return _LANG


def is_rtl() -> bool:
    return _LANG == "fa"


def tr(key: str, **fmt) -> str:
    entry = STRINGS.get(key)
    if entry is None:
        text = key
    else:
        text = entry[1] if _LANG == "fa" else entry[0]
        if not text:
            text = entry[0]
    if fmt:
        try:
            text = text.format(**fmt)
        except (KeyError, IndexError):
            pass
    return text
