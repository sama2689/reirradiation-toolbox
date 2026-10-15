from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import SimpleITK as sitk

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

Orientation = Literal["axial", "coronal", "sagittal"]
OverlayKind = Literal["mask", "dose"]
ComparisonMode = Literal["overlay", "difference"]
OverlayRenderMode = Literal["fill", "outline"]


@dataclass(slots=True)
class OverlayLayer:
    name: str
    image: sitk.Image
    kind: OverlayKind = "mask"
    color: tuple[int, int, int] = (255, 99, 71)
    opacity: float = 0.35
    visible: bool = True
    render_mode: OverlayRenderMode = "outline"


@dataclass(slots=True)
class _PreparedOverlay:
    name: str
    source_image: sitk.Image
    volume: np.ndarray
    kind: OverlayKind
    color: tuple[int, int, int]
    opacity: float
    visible: bool
    render_mode: OverlayRenderMode
    dose_max: float | None = None


class SquareCTDisplay(QWidget):
    """Fast CT viewer that keeps the source resolution and lets Qt handle final scaling.

    The expensive part is done once per slice change, not on every resize.
    """

    CONTOUR_OUTLINE_THICKNESS = 2

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.volume: np.ndarray | None = None
        self.spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._reference_image: sitk.Image | None = None
        self.current_slice = 0
        self.window = 1000
        self.level = 0
        self.orientation: Orientation = "axial"
        self._overlays: list[_PreparedOverlay] = []
        self._comparison_image: sitk.Image | None = None
        self._comparison_volume: np.ndarray | None = None
        self._comparison_difference_volume: np.ndarray | None = None
        self._comparison_mode: ComparisonMode = "overlay"
        self._comparison_opacity = 0.5
        self._comparison_hue_enabled = True
        self._comparison_base_color: tuple[int, int, int] = (253, 224, 71)
        self._comparison_color: tuple[int, int, int] = (59, 130, 246)
        self._comparison_diff_max = 0.0

        self.pixmap: QPixmap | None = None
        self.slice_exceeded = False
        self.render_scale_factor = 1.0
        self._render_target_size: QSize | None = None
        self._render_cache_key: tuple | None = None
        self._dose_scale_label = ""
        # Default contour outline thickness in pixels.
        self.outline_thickness = self.CONTOUR_OUTLINE_THICKNESS

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        width = 420
        return QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QSize:
        width = 240
        return QSize(width, self.heightForWidth(width))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        aspect_ratio = self._display_aspect_ratio()
        if aspect_ratio <= 0:
            return w
        return max(180, int(round(w / aspect_ratio)))

    def set_volume(self, image: sitk.Image | None) -> None:
        if image is None:
            self.volume = None
            self._reference_image = None
            self.spacing = (1.0, 1.0, 1.0)
            self.current_slice = 0
            self.slice_exceeded = False
            self.pixmap = None
            self._render_target_size = None
            self._render_cache_key = None
            self._dose_scale_label = ""
            self.update()
            return

        cast_image = sitk.Cast(image, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(cast_image).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        spacing = tuple(float(v) for v in cast_image.GetSpacing())
        if len(spacing) == 2:
            spacing = (spacing[0], spacing[1], 1.0)
        elif len(spacing) != 3:
            spacing = (1.0, 1.0, 1.0)

        self._reference_image = cast_image
        self.volume = arr
        self.spacing = spacing
        self.current_slice = 0
        self.slice_exceeded = False
        self._render_target_size = None
        self._render_cache_key = None
        self._refresh_prepared_overlays()
        self._refresh_prepared_comparison()
        self.update_display()

    def set_orientation(self, orientation: str) -> None:
        normalized = orientation.strip().lower()
        if normalized not in {"axial", "coronal", "sagittal"}:
            normalized = "axial"

        self.orientation = normalized  # type: ignore[assignment]
        self.update_display()

    def set_slice(self, index: int) -> None:
        if self.volume is None:
            return

        self.current_slice = int(index)
        self.update_display()

    def set_window_level(self, window: int, level: int) -> None:
        self.window = max(1, int(window))
        self.level = int(level)
        self.update_display()

    def set_render_scale_factor(self, factor: float) -> None:
        """Control the internal render resolution multiplier.

        The viewer still only renders the visible slice, but this lets the
        resampling path generate a denser intermediate image for smoother output
        on large or high-DPI displays.
        """
        self.render_scale_factor = max(1.0, float(factor))
        self.update_display()

    def set_overlays(self, layers: Iterable[OverlayLayer]) -> None:
        prepared: list[_PreparedOverlay] = []
        for layer in layers:
            prepared_volume = self._prepare_volume(layer.image, kind=layer.kind)
            prepared.append(
                _PreparedOverlay(
                    name=layer.name,
                    source_image=layer.image,
                    volume=prepared_volume,
                    kind=layer.kind,
                    color=layer.color,
                    opacity=float(np.clip(layer.opacity, 0.0, 1.0)),
                    visible=bool(layer.visible),
                    render_mode=layer.render_mode,
                    dose_max=self._image_max(layer.image) if layer.kind == "dose" else None,
                )
            )
        self._overlays = prepared
        self._refresh_prepared_overlays()
        self.update_display()

    def set_comparison_image(self, image: sitk.Image | None) -> None:
        self._comparison_image = image
        self._refresh_prepared_comparison()
        self.update_display()

    def set_comparison_mode(self, mode: str) -> None:
        normalized = mode.strip().lower()
        if normalized not in {"overlay", "difference"}:
            normalized = "overlay"
        self._comparison_mode = normalized  # type: ignore[assignment]
        self.update_display()

    def set_comparison_opacity(self, opacity: float) -> None:
        self._comparison_opacity = float(np.clip(opacity, 0.0, 1.0))
        self.update_display()

    def set_comparison_hue_enabled(self, enabled: bool) -> None:
        self._comparison_hue_enabled = bool(enabled)
        self.update_display()

    def set_comparison_base_color(self, color: tuple[int, int, int]) -> None:
        self._comparison_base_color = tuple(int(np.clip(v, 0, 255)) for v in color)
        self.update_display()

    def set_comparison_color(self, color: tuple[int, int, int]) -> None:
        self._comparison_color = tuple(int(np.clip(v, 0, 255)) for v in color)
        self.update_display()

    def set_comparison_properties(
        self,
        *,
        image: sitk.Image | None = None,
        mode: str | None = None,
        opacity: float | None = None,
        hue_enabled: bool | None = None,
        base_color: tuple[int, int, int] | None = None,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        if image is not None:
            self._comparison_image = image
            self._refresh_prepared_comparison()
        if mode is not None:
            self.set_comparison_mode(mode)
        if opacity is not None:
            self._comparison_opacity = float(np.clip(opacity, 0.0, 1.0))
        if hue_enabled is not None:
            self._comparison_hue_enabled = bool(hue_enabled)
        if base_color is not None:
            self._comparison_base_color = tuple(int(np.clip(v, 0, 255)) for v in base_color)
        if color is not None:
            self._comparison_color = tuple(int(np.clip(v, 0, 255)) for v in color)
        self.update_display()

    def clear_comparison_image(self) -> None:
        self._comparison_image = None
        self._comparison_volume = None
        self._comparison_difference_volume = None
        self._comparison_diff_max = 0.0
        self.update_display()

    def set_overlay_visible(self, name: str, visible: bool) -> None:
        for layer in self._overlays:
            if layer.name == name:
                layer.visible = visible
                self.update_display()
                return

    def clear_overlays(self) -> None:
        self._overlays.clear()
        self.update_display()

    def _refresh_prepared_overlays(self) -> None:
        if not self._overlays:
            return

        for overlay in self._overlays:
            overlay.volume = self._prepare_volume(overlay.source_image, kind=overlay.kind)

    def _refresh_prepared_comparison(self) -> None:
        if self._comparison_image is None:
            self._comparison_volume = None
            self._comparison_difference_volume = None
            self._comparison_diff_max = 0.0
            return

        self._comparison_volume = self._prepare_volume(self._comparison_image, kind="comparison")
        self._comparison_difference_volume = None
        self._comparison_diff_max = 0.0

        if self._reference_image is None:
            return

        try:
            fixed = sitk.Cast(self._reference_image, sitk.sitkFloat32)
            moving = sitk.Cast(self._comparison_image, sitk.sitkFloat32)
            if (
                fixed.GetSize() != moving.GetSize()
                or fixed.GetSpacing() != moving.GetSpacing()
                or fixed.GetOrigin() != moving.GetOrigin()
                or fixed.GetDirection() != moving.GetDirection()
            ):
                moving = sitk.Resample(
                    moving,
                    fixed,
                    sitk.Transform(),
                    sitk.sitkLinear,
                    0.0,
                    sitk.sitkFloat32,
                )

            difference = sitk.Abs(sitk.Subtract(fixed, moving))
            self._comparison_difference_volume = self._prepare_volume(difference, kind="comparison")

            diff_arr = sitk.GetArrayFromImage(difference).astype(np.float32)
            valid = np.isfinite(diff_arr)
            if np.any(valid):
                finite_values = diff_arr[valid]
                robust_max = float(np.percentile(finite_values, 99.0))
                if not np.isfinite(robust_max) or robust_max <= 0:
                    robust_max = float(np.max(finite_values))
                self._comparison_diff_max = max(robust_max, 1e-6)
        except Exception:
            self._comparison_difference_volume = None
            self._comparison_diff_max = 0.0

    def _content_rect(self) -> QRect:
        margin = 4
        return self.rect().adjusted(margin, margin, -margin, -margin)

    def update_display(self) -> None:
        if self.volume is None:
            self.pixmap = None
            self.slice_exceeded = False
            self._render_target_size = None
            self._render_cache_key = None
            self.update()
            return

        self.slice_exceeded = self._slice_out_of_range(self.current_slice)
        if self.slice_exceeded:
            self.pixmap = None
            self._render_target_size = None
            self._render_cache_key = None
            self.update()
            return

        content = self._content_rect()
        target = self._fit_display_rect(content)
        if target.width() <= 0 or target.height() <= 0:
            self.pixmap = None
            self._render_target_size = None
            self._render_cache_key = None
            self.update()
            return

        scale_factor = max(1.0, float(self.render_scale_factor))
        target_size = QSize(
            max(1, int(round(target.width() * scale_factor))),
            max(1, int(round(target.height() * scale_factor))),
        )
        render_key = (
            int(self.current_slice),
            self.orientation,
            int(target_size.width()),
            int(target_size.height()),
            int(self.window),
            int(self.level),
            float(scale_factor),
            tuple((layer.name, layer.visible, layer.kind, layer.opacity, layer.color, layer.render_mode) for layer in self._overlays),
            self._comparison_mode,
            float(self._comparison_opacity),
            bool(self._comparison_hue_enabled),
            tuple(int(v) for v in self._comparison_base_color),
            tuple(int(v) for v in self._comparison_color),
            float(self._comparison_diff_max),
        )
        if self._render_cache_key == render_key and self.pixmap is not None and not self.pixmap.isNull():
            self.update()
            return

        self.pixmap = self._render_slice_pixmap(target_size)
        self._render_target_size = target_size
        self._render_cache_key = render_key
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_display()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f8fafc"))

        content = self._content_rect()

        pen = QPen(QColor("#cbd5e1"))
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(content.adjusted(1, 1, -1, -1), 10, 10)

        if self.slice_exceeded:
            target = self._fit_display_rect(content)
            painter.fillRect(target, Qt.GlobalColor.black)
            painter.setPen(QColor("white"))
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(target, Qt.AlignmentFlag.AlignCenter, "Slice range exceeded.")
            return

        if self.pixmap is None or self.pixmap.isNull():
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(content, Qt.AlignmentFlag.AlignCenter, "No image loaded.")
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = self._fit_display_rect(content)
        painter.drawPixmap(target, self.pixmap)
        self._draw_legend(painter, target.x(), target.y(), target.width(), target.height())
        self._draw_dose_scale(painter, target)

    def _draw_legend(self, painter: QPainter, x: int, y: int, width: int, height: int) -> None:
        if any(layer.visible and layer.kind == "dose" for layer in self._overlays):
            return

        legend_entries = [layer for layer in self._overlays if layer.visible and layer.kind != "dose"]
        if not legend_entries:
            return

        padding = 8
        swatch = 10
        line_height = 18
        text_width = 0
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        metrics = painter.fontMetrics()
        for entry in legend_entries:
            text_width = max(text_width, metrics.horizontalAdvance(self._legend_label(entry.name)))

        box_width = min(width - 20, text_width + swatch + 4 * padding)
        box_height = min(height - 20, len(legend_entries) * line_height + 2 * padding)
        box_x = x + 10
        box_y = y + 10

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 160))
        painter.drawRoundedRect(box_x, box_y, box_width, box_height, 8, 8)

        for index, entry in enumerate(legend_entries):
            row_y = box_y + padding + index * line_height
            if row_y > box_y + box_height - line_height:
                break

            color = QColor(*entry.color)
            painter.setBrush(color)
            painter.drawRoundedRect(box_x + padding, row_y + 4, swatch, swatch, 2, 2)
            painter.setPen(QColor("white"))
            painter.drawText(
                box_x + padding + swatch + 8,
                row_y + 13,
                self._legend_label(entry.name),
            )

    def _fit_display_rect(self, content: QRect) -> QRect:
        content_w = content.width()
        content_h = content.height()
        if content_w <= 0 or content_h <= 0:
            return content

        aspect = self._display_aspect_ratio()
        if aspect <= 0:
            return content

        content_aspect = content_w / content_h
        if content_aspect >= aspect:
            draw_h = content_h
            draw_w = max(1, int(round(draw_h * aspect)))
        else:
            draw_w = content_w
            draw_h = max(1, int(round(draw_w / aspect)))

        x = content.x() + (content_w - draw_w) // 2
        y = content.y() + (content_h - draw_h) // 2
        return QRect(x, y, draw_w, draw_h)

    def _display_aspect_ratio(self) -> float:
        if self.volume is None:
            return 1.0

        rows, cols = self._current_slice_shape()
        row_spacing, col_spacing = self._orientation_spacing()
        physical_height = max(rows * row_spacing, 1e-6)
        physical_width = max(cols * col_spacing, 1e-6)
        return physical_width / physical_height

    def _current_slice_shape(self) -> tuple[int, int]:
        if self.volume is None:
            return 1, 1

        if self.orientation == "coronal":
            return self.volume.shape[0], self.volume.shape[2]
        if self.orientation == "sagittal":
            return self.volume.shape[0], self.volume.shape[1]
        return self.volume.shape[1], self.volume.shape[2]

    def _prepare_volume(self, image: sitk.Image, *, kind: str = "mask") -> np.ndarray:
        prepared_image = self._resample_to_reference(image, is_mask=(kind == "mask"))
        array = sitk.GetArrayFromImage(prepared_image).astype(np.float32)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        return array

    def _resample_volume_to_isotropic(self, image: sitk.Image) -> tuple[sitk.Image, tuple[float, float, float]]:
        if image.GetDimension() != 3:
            cast_image = sitk.Cast(image, sitk.sitkFloat32)
            spacing = tuple(float(v) for v in cast_image.GetSpacing())
            if len(spacing) == 2:
                spacing = (spacing[0], spacing[1], 1.0)
            elif len(spacing) != 3:
                spacing = (1.0, 1.0, 1.0)
            return cast_image, spacing

        spacing = tuple(float(v) for v in image.GetSpacing())
        if len(spacing) != 3:
            spacing = (1.0, 1.0, 1.0)

        target_spacing = float(min(spacing))
        if target_spacing <= 0:
            target_spacing = 1.0

        size = np.asarray(image.GetSize(), dtype=np.int64)
        spacing_arr = np.asarray(spacing, dtype=np.float64)
        target_spacing_arr = np.asarray((target_spacing, target_spacing, target_spacing), dtype=np.float64)
        new_size = np.maximum(
            np.rint(size * spacing_arr / target_spacing_arr).astype(np.int64),
            1,
        )

        if np.allclose(spacing_arr, target_spacing_arr, atol=1e-6, rtol=0.0):
            return sitk.Cast(image, sitk.sitkFloat32), (target_spacing, target_spacing, target_spacing)

        resampled = sitk.Resample(
            sitk.Cast(image, sitk.sitkFloat32),
            [int(v) for v in new_size.tolist()],
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            (target_spacing, target_spacing, target_spacing),
            image.GetDirection(),
            0.0,
            sitk.sitkFloat32,
        )
        return resampled, (target_spacing, target_spacing, target_spacing)

    def _resample_to_reference(self, image: sitk.Image, *, is_mask: bool) -> sitk.Image:
        if self._reference_image is None:
            return sitk.Cast(image, sitk.sitkFloat32)

        interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        return sitk.Resample(
            sitk.Cast(image, sitk.sitkFloat32),
            self._reference_image,
            sitk.Transform(),
            interpolator,
            0.0,
            sitk.sitkFloat32,
        )

    def _clamp_slice(self, index: int) -> int:
        if self.volume is None:
            return 0

        max_index = self._axis_length(self.volume) - 1
        if max_index < 0:
            return 0
        return max(0, min(int(index), max_index))

    def _axis_length(self, volume: np.ndarray) -> int:
        if self.orientation == "coronal":
            return volume.shape[1]
        if self.orientation == "sagittal":
            return volume.shape[2]
        return volume.shape[0]

    def _extract_slice(self, volume: np.ndarray, index: int) -> np.ndarray:
        if self.orientation == "coronal":
            slice_2d = volume[:, index, :]
            return np.flipud(np.asarray(slice_2d, dtype=np.float32))
        if self.orientation == "sagittal":
            slice_2d = volume[:, :, index]
            return np.flipud(np.asarray(slice_2d, dtype=np.float32))

        slice_2d = volume[index, :, :]
        return np.asarray(slice_2d, dtype=np.float32)

    def _orientation_spacing(self) -> tuple[float, float]:
        if self.orientation == "coronal":
            return max(self.spacing[2], 1e-6), max(self.spacing[0], 1e-6)
        if self.orientation == "sagittal":
            return max(self.spacing[2], 1e-6), max(self.spacing[1], 1e-6)
        return max(self.spacing[1], 1e-6), max(self.spacing[0], 1e-6)

    def _grayscale_rgba(self, slice_array: np.ndarray) -> np.ndarray:
        low = self.level - self.window / 2
        high = self.level + self.window / 2
        if high <= low:
            high = low + 1

        norm = (slice_array - low) / (high - low)
        norm = np.clip(norm, 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)

        rgba = np.empty((*gray.shape, 4), dtype=np.uint8)
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = 255
        return rgba

    def _windowed_normalized(self, slice_array: np.ndarray) -> np.ndarray:
        low = self.level - self.window / 2
        high = self.level + self.window / 2
        if high <= low:
            high = low + 1
        norm = (np.asarray(slice_array, dtype=np.float32) - low) / (high - low)
        return np.clip(norm, 0.0, 1.0)

    def _difference_rgba(self, slice_array: np.ndarray, max_value: float | None = None) -> np.ndarray:
        values = np.asarray(slice_array, dtype=np.float32)
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)

        finite = np.isfinite(values)
        if not np.any(finite):
            return rgba

        finite_values = values[finite]
        max_val = max_value if max_value is not None else float(np.percentile(finite_values, 99.0))
        if not np.isfinite(max_val) or max_val <= 0:
            max_val = float(np.max(finite_values)) if finite_values.size else 1.0
        max_val = max(max_val, 1e-6)

        norm = np.clip(values / max_val, 0.0, 1.0)
        # Keep the difference map dark by default while still showing subtle error.
        norm = np.power(norm, 0.85)
        gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        gray[~finite] = 0

        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = 255
        return rgba

    def _comparison_rgba(
        self,
        slice_array: np.ndarray,
        color: tuple[int, int, int],
        *,
        opacity: float | None = None,
        force_grayscale: bool = False,
    ) -> np.ndarray:
        values = np.asarray(slice_array, dtype=np.float32)
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)

        if not np.any(np.isfinite(values)):
            return rgba

        low = self.level - self.window / 2
        high = self.level + self.window / 2
        if high <= low:
            high = low + 1

        norm = np.clip((values - low) / (high - low), 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)

        if force_grayscale:
            rgba[..., 0] = gray
            rgba[..., 1] = gray
            rgba[..., 2] = gray
        else:
            color_arr = np.array(color, dtype=np.float32) / 255.0
            rgb = (gray[..., None].astype(np.float32) / 255.0) * color_arr[None, None, :]
            rgba[..., :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(255.0 * (self._comparison_opacity if opacity is None else opacity), 0, 255).astype(np.uint8)
        return rgba

    def _overlay_rgba(self, slice_array: np.ndarray, overlay: _PreparedOverlay) -> np.ndarray:
        rgba = np.zeros((*slice_array.shape, 4), dtype=np.uint8)

        if overlay.kind == "dose":
            values = np.asarray(slice_array, dtype=np.float32)
            valid = np.isfinite(values)
            if not np.any(valid):
                return rgba

            finite_values = values[valid]
            low = 0.0
            high = overlay.dose_max if overlay.dose_max is not None else float(np.max(finite_values))
            if not np.isfinite(high) or high <= low:
                high = low + 1.0

            norm = np.clip((values - low) / (high - low), 0.0, 1.0)
            rgba[..., :3] = self._dose_colormap(norm)
            rgba[..., 3] = np.clip(norm * 255.0 * overlay.opacity * 0.82, 0, 255).astype(np.uint8)
            return rgba

        mask = slice_array > 0
        if overlay.render_mode == "outline":
            mask = self._outline_mask(mask, thickness=self.outline_thickness)

        if not np.any(mask):
            return rgba

        rgba[mask, 0] = overlay.color[0]
        rgba[mask, 1] = overlay.color[1]
        rgba[mask, 2] = overlay.color[2]
        rgba[mask, 3] = int(255 * overlay.opacity)
        return rgba

    @staticmethod
    def _alpha_composite(base_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        # Premultiplied alpha math in float32 keeps it clear and fast enough at source resolution.
        base = base_rgba.astype(np.float32) / 255.0
        over = overlay_rgba.astype(np.float32) / 255.0

        over_a = over[..., 3:4]
        base_a = base[..., 3:4]

        out_a = over_a + base_a * (1.0 - over_a)
        denom = np.maximum(out_a, 1e-6)
        out_rgb = (over[..., :3] * over_a + base[..., :3] * base_a * (1.0 - over_a)) / denom

        out = np.empty_like(base_rgba, dtype=np.uint8)
        out[..., :3] = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)
        return out

    def _outline_mask(self, mask: np.ndarray, thickness: int | None = None) -> np.ndarray:
        if thickness is None:
            thickness = self.outline_thickness

        thickness = max(1, int(thickness))

        if mask.ndim != 2 or not np.any(mask):
            return mask

        current = mask.astype(bool, copy=False)

        def erode_once(source: np.ndarray) -> np.ndarray:
            padded = np.pad(source, 1, mode="constant", constant_values=False)
            h, w = source.shape
            neighbors = [
                padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            ]
            return np.logical_and.reduce(neighbors)

        eroded = current
        for _ in range(thickness):
            eroded = erode_once(eroded)
            if not np.any(eroded):
                break

        # Keep the contour tight: no post-dilation, just the morphological edge.
        return current & ~eroded

    def _slice_out_of_range(self, index: int) -> bool:
        if self.volume is None:
            return False
        axis_len = self._axis_length(self.volume)
        return index < 0 or index >= axis_len

    def _render_slice_pixmap(self, target_size: QSize) -> QPixmap:
        assert self.volume is not None
        target_w = max(1, int(target_size.width()))
        target_h = max(1, int(target_size.height()))

        base_slice = self._extract_slice(self.volume, self.current_slice)
        base_slice = self._resample_2d_slice(base_slice, target_w, target_h, continuous=True)

        if self._comparison_mode == "difference" and self._comparison_difference_volume is not None:
            comparison_slice = self._extract_slice(self._comparison_difference_volume, self.current_slice)
            comparison_slice = self._resample_2d_slice(comparison_slice, target_w, target_h, continuous=True)
            diff_rgba = self._difference_rgba(comparison_slice, max_value=self._comparison_diff_max)
            image = self._rgba_to_qimage(diff_rgba)
            return QPixmap.fromImage(image)

        if self._comparison_hue_enabled and self._comparison_volume is not None:
            base_rgba = self._comparison_rgba(base_slice, self._comparison_base_color, opacity=1.0)
        else:
            base_rgba = self._grayscale_rgba(base_slice)

        if self._comparison_volume is not None:
            comparison_slice = self._extract_slice(self._comparison_volume, self.current_slice)
            comparison_slice = self._resample_2d_slice(comparison_slice, target_w, target_h, continuous=True)
            if comparison_slice.shape == base_slice.shape:
                if self._comparison_hue_enabled:
                    comparison_rgba = self._comparison_rgba(comparison_slice, self._comparison_color, opacity=self._comparison_opacity)
                else:
                    comparison_rgba = self._comparison_rgba(comparison_slice, (255, 255, 255), opacity=self._comparison_opacity, force_grayscale=True)
                base_rgba = self._alpha_composite(base_rgba, comparison_rgba)

        for overlay in self._overlays:
            if not overlay.visible:
                continue

            overlay_slice = self._extract_slice(overlay.volume, self.current_slice)
            overlay_slice = self._resample_2d_slice(
                overlay_slice,
                target_w,
                target_h,
                continuous=(overlay.kind == "dose"),
                is_mask=(overlay.kind == "mask"),
            )
            if overlay_slice.shape != base_slice.shape:
                continue

            overlay_rgba = self._overlay_rgba(overlay_slice, overlay)
            base_rgba = self._alpha_composite(base_rgba, overlay_rgba)

        image = self._rgba_to_qimage(base_rgba)
        return QPixmap.fromImage(image)

    def _resample_2d_slice(
        self,
        slice_array: np.ndarray,
        target_w: int,
        target_h: int,
        *,
        continuous: bool,
        is_mask: bool = False,
    ) -> np.ndarray:
        array = np.asarray(slice_array, dtype=np.float32)
        if array.ndim != 2 or array.size == 0:
            return np.zeros((max(1, target_h), max(1, target_w)), dtype=np.float32)

        if array.shape == (target_h, target_w):
            return array

        image = sitk.GetImageFromArray(array)
        row_spacing, col_spacing = self._orientation_spacing()
        image.SetSpacing((float(col_spacing), float(row_spacing)))

        interp = sitk.sitkNearestNeighbor if is_mask else (sitk.sitkBSpline if continuous else sitk.sitkLinear)
        output_spacing = (
            float(image.GetSpacing()[0]) * max(image.GetSize()[0] - 1, 1) / max(target_w - 1, 1),
            float(image.GetSpacing()[1]) * max(image.GetSize()[1] - 1, 1) / max(target_h - 1, 1),
        )
        resampled = sitk.Resample(
            sitk.Cast(image, sitk.sitkFloat32),
            [int(target_w), int(target_h)],
            sitk.Transform(),
            interp,
            image.GetOrigin(),
            output_spacing,
            image.GetDirection(),
            0.0,
            sitk.sitkFloat32,
        )
        return sitk.GetArrayFromImage(resampled).astype(np.float32)

    def _dose_colormap(self, norm: np.ndarray) -> np.ndarray:
        stops = np.array([
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ], dtype=np.float32)
        positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        rgb = np.empty((*norm.shape, 3), dtype=np.uint8)
        for channel in range(3):
            values = np.interp(norm, positions, stops[:, channel])
            rgb[..., channel] = np.clip(values * 255.0, 0, 255).astype(np.uint8)
        return rgb

    @staticmethod
    def _array_max(array: np.ndarray) -> float:
        values = np.asarray(array, dtype=np.float32)
        valid = np.isfinite(values)
        if not np.any(valid):
            return 0.0
        return float(np.max(values[valid]))

    @staticmethod
    def _image_max(image: sitk.Image) -> float:
        array = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32)).astype(np.float32)
        return SquareCTDisplay._array_max(array)

    @staticmethod
    def _format_dose_value(value: float) -> str:
        if not np.isfinite(value):
            return "0.00"
        abs_value = abs(float(value))
        if abs_value < 0.1:
            return f"{value:.3f}"
        if abs_value < 10.0:
            return f"{value:.2f}"
        return f"{value:.1f}"

    def _dose_scale_info(self) -> tuple[str, float]:
        dose_layers = [layer for layer in self._overlays if layer.visible and layer.kind == "dose"]
        if not dose_layers:
            return "", 0.0

        layer = dose_layers[-1]
        return self._legend_label(layer.name), float(layer.dose_max or 0.0)

    def _draw_dose_scale(self, painter: QPainter, target: QRect) -> None:
        label, max_value = self._dose_scale_info()
        if not label:
            return

        margin = 12
        bar_w = 14
        bar_h = min(128, max(72, target.height() // 4))
        value_gap = 8
        top_pad = 4
        label_gap = 6
        label_h = 12
        bottom_pad = 4
        panel_w = 182
        panel_h = top_pad + bar_h + label_gap + label_h + bottom_pad
        x = target.x() + margin
        y = target.y() + target.height() - margin - panel_h

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 145))
        painter.drawRoundedRect(x - 8, y - 6, panel_w, panel_h, 8, 8)

        bar_y = y + top_pad
        bar_x = x
        bar_bottom = bar_y + bar_h
        gradient = QLinearGradient(bar_x, bar_bottom, bar_x, bar_y)
        gradient.setColorAt(0.0, QColor(0, 0, 255))
        gradient.setColorAt(0.25, QColor(0, 255, 255))
        gradient.setColorAt(0.5, QColor(0, 255, 0))
        gradient.setColorAt(0.75, QColor(255, 255, 0))
        gradient.setColorAt(1.0, QColor(255, 0, 0))
        painter.setBrush(gradient)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)

        painter.setPen(QColor("white"))
        font = painter.font()
        font.setPointSize(7)
        font.setBold(False)
        painter.setFont(font)
        unit = "Gy"
        max_text = f"{self._format_dose_value(max_value)} {unit}"
        min_text = f"0.00 {unit}"
        text_x = x + bar_w + value_gap
        painter.drawText(text_x, bar_y + 10, max_text)
        painter.drawText(text_x, bar_y + bar_h, min_text)

        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        label_rect = QRect(x, bar_bottom + label_gap, panel_w - 16, label_h)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        painter.restore()

    def _rgba_to_qimage(self, rgba: np.ndarray) -> QImage:
        height, width, _channels = rgba.shape
        qimage = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format.Format_RGBA8888)
        return qimage.copy()

    @staticmethod
    def _legend_label(name: str) -> str:
        parts = name.replace("::", " ").replace("_", " ").split()
        return " ".join(parts)