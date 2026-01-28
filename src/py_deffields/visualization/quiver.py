# ruff: noqa: F722
from __future__ import annotations

import enum
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import einops

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from jaxtyping import Float, Inexact, Array


class AxisState(enum.Enum):
    """Enumeration of axis orderings for 3D data."""
    ZYX = "zyx"
    ZXY = "zxy"
    YXZ = "yxz"
    YZX = "yzx"
    XZY = "xzy"
    XYZ = "xyz"


def generate_einops_pattern(
    target_state: AxisState
) -> str:
    """
    Generate einops rearrangement pattern from ZYX to target axis order.
    """
    source_state: AxisState = AxisState.ZYX
    expanded_source_state = " ".join(source_state.value)
    expanded_target_state = " ".join(target_state.value)
    pattern = f"{expanded_source_state} -> {expanded_target_state}"
    return pattern


def _make_axis_dropdown():
    """
    Axis order dropdown.
    Initial state is assumed as ZYX (depth, height, width).
    """
    initial_axis_state = AxisState.ZYX
    return widgets.Dropdown(
        options=[axis_state.value for axis_state in AxisState],
        value=initial_axis_state.value,
        description='Axis Order:',
    )


def _make_subsample_factor_slider() -> widgets.IntSlider:
    """Subsample factor slider for in-plane quiver visualization."""
    return widgets.IntSlider(
        value=2,
        min=1,
        max=20,
        step=1,
        description="Subsample Factor:",
        continuous_update=False,
    )


def _make_axis_index_slider(
    axis_size: int,
) -> widgets.IntSlider:
    """Axis index slider for selecting slice in quiver visualization."""
    return widgets.IntSlider(
        value=axis_size // 2,
        min=0,
        max=axis_size - 1,
        step=1,
        description="Slice Index:",
        continuous_update=True,
    )



class QuiverVisualizer:
    """
    Visualize 3D deformation fields using quiver plots.
    """
    def __init__(
        self,
        fig: Figure,
        ax: Axes,
        deformation_field: Inexact[Array, "D H W 3"],
        axis_index_slider: widgets.IntSlider,
        subsample_factor_slider: widgets.IntSlider,
        axis_order_dropdown: widgets.Dropdown,
    ) -> None:
        """
        Initialize QuiverVisualizer with a 3D deformation field.

        Parameters
        ----------
        deformation_field : Inexact[Array, "D H W 3"]
            3D deformation field to visualize.
        """
        # this is the basic input deformation field in D H W 3 format
        self._input_deformation_field = deformation_field
        self._axis_index_slider = axis_index_slider
        self._subsample_factor_slider = subsample_factor_slider
        self._axis_order_dropdown = axis_order_dropdown
        self._ax = ax


    @classmethod
    def create(
        cls,
        deformation_field: Inexact[Array, "D H W 3"],
        ax: Axes | None = None,
    ) -> QuiverVisualizer:
        """
        Factory method to create QuiverVisualizer with default UI components.

        Parameters
        ----------
        deformation_field : Inexact[Array, "D H W 3"]
            3D deformation field to visualize.
        
        Returns
        -------
        QuiverVisualizer
            Initialized QuiverVisualizer instance.
        """
        if ax is None:
            fig, ax = plt.subplots()

        axis_order_dropdown = _make_axis_dropdown()
        subsample_factor_slider = _make_subsample_factor_slider()
        axis_index_slider = _make_axis_index_slider(deformation_field.shape[0])


    def query_deformation_field(self) -> Inexact[Array, "D H W 3"]:
        """
        Query the deformation field in the given axis state and subsample factor.
        """
        target_axis_state = AxisState(self._axis_order_dropdown.value)
        einops_pattern = generate_einops_pattern(target_axis_state)
        subsample_index = self._query_subsample_index_tuple()

        rearranged_field = einops.rearrange(
            self._input_deformation_field,
            einops_pattern
        )
        subsampled_field = rearranged_field[:, *subsample_index]
        return subsampled_field
        
        
    def _query_subsample_index_tuple(self) -> tuple[slice, slice]:
        """
        Get the subsample index tuple that returns a subsampled 2D plane
        from the 3D deformation field based on the current subsample factor.
        """
        subsample_factor: int = self._subsample_factor_slider.value
        plane_slice = tuple((
            slice(None, None, subsample_factor),
            slice(None, None, subsample_factor),
            )
        )
        return plane_slice 


    def _make_axis_index_slider(self) -> widgets.IntSlider:
        # we quey the deformation field in dependence of other UI settings

        deformation_field = self.query_deformation_field()
        max_index = deformation_field.shape[0] - 1
        
        if self._axis_index_slider is not None:
            value = min(self._axis_index_slider.value, max_index)
        else:
            value = max_index // 2

        return widgets.IntSlider(
            value=value,
            min=0,
            max=max_index,
            step=1,
            description="Slice Index:",
            continuous_update=True,
        )


    @staticmethod
    def _make_subsample_factor_slider() -> widgets.IntSlider:
        return widgets.IntSlider(
            value=2,
            min=1,
            max=20,
            step=1,
            description="Subsample Factor:",
            continuous_update=False,
        )