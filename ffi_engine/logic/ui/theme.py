# ui/theme.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Colors:
    # Base / Layout
    app_bg: str = "#F8FAFC"
    card_bg: str = "#FFFFFF"
    border: str = "#E5E7EB"

    # Text hierarchy
    text_h1: str = "#0F172A"       # Page title / strongest
    text_kpi_value: str = "#0F172A"
    text_h2: str = "#1F2937"       # Section / card titles
    text_h3: str = "#1F2937"
    text_body: str = "#3A454B"     # Reading comfort
    text_muted: str = "#64748B"    # Metadata / labels / quotes

    # State / Status (minimal use)
    status_success_bg: str = "#ECFDF3"
    status_warning_bg: str = "#FFFBEB"
    status_negative_bg: str = "#FEF2F2"

    # Optional: status text (keep calm)
    status_success_text: str = "#166534"  # subtle green
    status_warning_text: str = "#92400E"  # subtle amber
    status_negative_text: str = "#991B1B" # subtle red

@dataclass(frozen=True)
class Typography:
    font_family: str = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"

    # H1
    h1_size: int = 20
    h1_lh: int = 28
    h1_weight: int = 600
    h1_letter_spacing: str = "0"

    # H2
    h2_size: int = 16
    h2_lh: int = 24
    h2_weight: int = 600
    h2_letter_spacing: str = "0"

    # H3
    h3_size: int = 14
    h3_lh: int = 22
    h3_weight: int = 400
    h3_letter_spacing: str = "0"

    # Body
    body_size: int = 14
    body_lh: int = 22         # choose 22 as default for Nordic reading comfort
    body_weight: int = 400
    body_letter_spacing: str = "0"

    # KPI label/header
    kpi_label_size: int = 12
    kpi_label_lh: int = 16
    kpi_label_weight: int = 500
    kpi_label_letter_spacing: str = "0.01em"

    # KPI value
    kpi_value_size: int = 18
    kpi_value_lh: int = 24
    kpi_value_weight: int = 600
    kpi_value_letter_spacing: str = "0"

    # Quote / explanatory
    quote_size: int = 14
    quote_lh: int = 22
    quote_weight: int = 400
    quote_letter_spacing: str = "0.01em"

@dataclass(frozen=True)
class Layout:
    max_width_px: int = 1100
    page_pad_top_px: int = 24
    page_pad_bottom_px: int = 48

    section_gap_px: int = 32
    card_pad_px: int = 20
    card_radius_px: int = 12

    # internal spacing defaults
    inline_gap_px: int = 16
    tight_gap_px: int = 8

@dataclass(frozen=True)
class Navigator:
    bg: str = "#F8FAFC"
    active_bg: str = "#E5E7EB"
    active_text: str = "#1F2937"
    font_family: str = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"
    font_weight: int = 500
    font_size: int = 14

COLORS = Colors()
TYPE = Typography()
LAYOUT = Layout()
NAV = Navigator()
