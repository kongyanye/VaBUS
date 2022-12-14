import copy
import logging

from .base import Base
from ..annotation import Annotation
from .. import headmeta
from ..show.painters import AnnotationPainter, KeypointPainter
from ..show.fields import boxes, margins, quiver

CMAP_ORANGES_NAN = None

LOG = logging.getLogger(__name__)


class Cif(Base):
    show_margin = False
    show_confidences = False
    show_regressions = False
    show_background = False

    def __init__(self, meta: headmeta.Cif):
        super().__init__(meta.name)
        self.meta = meta
        keypoint_painter = KeypointPainter(monocolor_connections=True)
        self.annotation_painter = AnnotationPainter(
            keypoint_painter=keypoint_painter)

    def targets(self, field, *, annotation_dicts):
        assert self.meta.keypoints is not None
        assert self.meta.draw_skeleton is not None

        annotations = [
            Annotation(keypoints=self.meta.keypoints,
                       skeleton=self.meta.draw_skeleton,
                       sigmas=self.meta.sigmas,
                       score_weights=self.meta.score_weights).set(
                           ann['keypoints'],
                           fixed_score='',
                           fixed_bbox=ann['bbox']) for ann in annotation_dicts
        ]

        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 4], annotations=annotations)

    def predicted(self, field):
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3],
                          field[:, 4],
                          annotations=self._ground_truth,
                          confidence_fields=field[:, 0],
                          uv_is_offset=False)

    def _confidences(self, confidences):
        if not self.show_confidences:
            return

        for f in self.indices:
            LOG.debug('%s', self.meta.keypoints[f])

            with self.image_canvas(self._processed_image,
                                   margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f],
                                                 self.meta.stride),
                               alpha=0.9,
                               vmin=0.0,
                               vmax=1.0,
                               cmap=CMAP_ORANGES_NAN)
                self.colorbar(ax, im)

    def _regressions(self,
                     regression_fields,
                     scale_fields,
                     *,
                     annotations=None,
                     confidence_fields=None,
                     uv_is_offset=True):
        if not self.show_regressions:
            return

        for f in self.indices:
            LOG.debug('%s', self.meta.keypoints[f])
            confidence_field = confidence_fields[
                f] if confidence_fields is not None else None

            with self.image_canvas(self._processed_image,
                                   margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                white_screen(ax, alpha=0.5)
                if annotations:
                    self.annotation_painter.annotations(ax,
                                                        annotations,
                                                        color='lightgray')
                q = quiver(ax,
                           regression_fields[f, :2],
                           confidence_field=confidence_field,
                           xy_scale=self.meta.stride,
                           uv_is_offset=uv_is_offset,
                           cmap='Oranges',
                           clim=(0.5, 1.0),
                           width=0.001)
                boxes(ax,
                      scale_fields[f] / 2.0,
                      confidence_field=confidence_field,
                      regression_field=regression_fields[f, :2],
                      xy_scale=self.meta.stride,
                      cmap='Oranges',
                      fill=False,
                      regression_field_is_offset=uv_is_offset)
                if self.show_margin:
                    margins(ax,
                            regression_fields[f, :6],
                            xy_scale=self.meta.stride)

                self.colorbar(ax, q)
