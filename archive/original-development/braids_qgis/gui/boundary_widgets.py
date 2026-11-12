"""Structured per-ID hydraulic and sediment boundary controls."""
from __future__ import annotations

from dataclasses import asdict

from qgis.PyQt.QtWidgets import (QComboBox,QDoubleSpinBox,QFormLayout,QHBoxLayout,
    QLabel,QPlainTextEdit,QPushButton,QVBoxLayout)

from braids_core.boundaries.cut_cell_io import CutHydraulicBoundarySeries,CutSedimentBoundarySeries
from .accordion import AccordionSection


def _spin(minimum,maximum,value,decimals=6,step=.1):
    widget=QDoubleSpinBox();widget.setRange(minimum,maximum);widget.setDecimals(decimals)
    widget.setValue(value);widget.setSingleStep(step);return widget


class HydraulicBoundaryCard(AccordionSection):
    """The currently executable hydraulic settings for one logical face ID."""
    def __init__(self,bc_id,kind,values,on_change,on_select,on_preview,on_plot,on_delete,
                 *,expanded=False,parent=None):
        super().__init__(f"{kind.capitalize()} · {bc_id}",expanded=expanded,parent=parent)
        self.bc_id=bc_id;self.kind=kind;self._on_change=on_change
        form=QFormLayout();self.body_layout.addLayout(form)
        self.mode=QComboBox()
        if kind=="inlet":
            self.mode.addItem("Total discharge (free water level)","total_discharge")
            self.mode.addItem("Legacy discharge + water level","discharge")
            self.mode.addItem("Water level + discharge","inlet_stage")
            self.mode.addItem("Water depth + discharge","inlet_depth")
            self.mode.addItem("Normal depth + discharge","inlet_normal")
            self.mode.addItem("Critical depth + discharge","inlet_critical")
        else:
            self.mode.addItem("Water-level outlet","stage");self.mode.addItem("Critical-depth outlet","critical")
            self.mode.addItem("Free weir outlet","weir");self.mode.addItem("Supercritical outflow","supercritical")
        self.forcing=QComboBox();self.forcing.addItem("Constant","constant");self.forcing.addItem("Time series","series")
        self.profile=QComboBox();self.profile.addItem("Stock conveyance","stock_conveyance")
        # Keep the historical persisted token for old projects, but expose the
        # generalized C/n/ks physical operator rather than calling it Manning.
        self.profile.addItem("Viscosity-equilibrated roughness profile","viscous_manning_equilibrium")
        self.profile.setToolTip(
            "Solves the selected Chezy, Manning, or White-Colebrook transverse equilibrium profile without converting the raw roughness parameter.")
        self.discharge=_spin(0.,1e12,1.,6,.1);self.stage=_spin(-1e7,1e7,0.,6,.1)
        self.depth=_spin(0.,1e7,1.,6,.1)
        self._energy_slope=float(values.get("energy_slope",.001))
        self._boundary_manning=values.get("boundary_manning_n")
        self.weir_coefficient=_spin(1e-12,1e4,1.7,6,.1);self.weir_width=_spin(0.,1e7,0.,6,.1)
        form.addRow("Boundary mode",self.mode);form.addRow("Forcing",self.forcing)
        form.addRow("Total-Q inlet profile",self.profile)
        form.addRow("Total discharge (m³/s)",self.discharge);form.addRow("Water level / crest (m)",self.stage)
        form.addRow("Water depth above datum (m)",self.depth);form.addRow("Weir coefficient (m½/s)",self.weir_coefficient)
        form.addRow("Weir width (m; 0 = selected length)",self.weir_width)
        self.series_label=QLabel();self.series_label.setWordWrap(True)
        self.series=QPlainTextEdit();self.series.setMinimumHeight(105)
        self.body_layout.addWidget(self.series_label);self.body_layout.addWidget(self.series)
        self.summary=QLabel();self.summary.setWordWrap(True);self.body_layout.addWidget(self.summary)
        buttons=QHBoxLayout();self.select_button=QPushButton("Highlight Assigned Face" if kind=="inlet" else "Highlight Assigned Faces");self.preview_button=QPushButton("Preview Section");self.plot_button=QPushButton("Plot Forcing");self.delete_button=QPushButton("Delete BC")
        self.select_button.setToolTip("Selects and zooms to the polygon face(s) already assigned to this boundary ID. It does not change the assignment.")
        self.preview_button.setToolTip("Shows the terrain cross-section and the part below the resolved horizontal water level. This wetted section is where boundary flow can pass.")
        self.delete_button.setToolTip("Deletes this boundary condition and returns all of its assigned polygon faces to walls.")
        buttons.addWidget(self.select_button);buttons.addWidget(self.preview_button);buttons.addWidget(self.plot_button);buttons.addWidget(self.delete_button);self.body_layout.addLayout(buttons)
        self.plot_button.setVisible(kind=="inlet")
        self.select_button.clicked.connect(lambda:on_select(self));self.preview_button.clicked.connect(lambda:on_preview(self));self.plot_button.clicked.connect(lambda:on_plot(self));self.delete_button.clicked.connect(lambda:on_delete(self))
        self.set_values(values)
        self.mode.currentIndexChanged.connect(self._changed);self.forcing.currentIndexChanged.connect(self._changed)
        self.profile.currentIndexChanged.connect(self._changed)
        for widget in (self.discharge,self.stage,self.depth,self.weir_coefficient,self.weir_width):widget.valueChanged.connect(self._changed)
        self.series.textChanged.connect(self._changed)

    def _changed(self,*_):
        mode=self.mode.currentData();inlet=self.kind=="inlet"
        schedule_supported=mode in ("total_discharge","discharge","inlet_stage","inlet_depth","inlet_normal","inlet_critical","stage")
        if not schedule_supported and self.forcing.currentData()=="series":self.forcing.setCurrentIndex(self.forcing.findData("constant"))
        self.forcing.setEnabled(schedule_supported);series=schedule_supported and self.forcing.currentData()=="series"
        self.profile.setVisible(mode=="total_discharge")
        profile_label=self.body_layout.itemAt(0).layout().labelForField(self.profile)
        if profile_label is not None:profile_label.setVisible(mode=="total_discharge")
        enabled={self.discharge:inlet and not series,self.stage:mode in ("discharge","inlet_stage","stage","weir") and not series,self.depth:mode=="inlet_depth" and not series,
            self.weir_coefficient:mode=="weir",self.weir_width:mode=="weir"}
        for widget,state in enabled.items():
            widget.setEnabled(state);form=self.body_layout.itemAt(0).layout();label=form.labelForField(widget)
            widget.setVisible(state)
            if label is not None:label.setVisible(state)
        if mode in ("discharge","inlet_stage","inlet_depth"):
            control="water_depth_m" if mode=="inlet_depth" else "water_level_m"
            self.series_label.setText(f"One row per knot: time_s, total_discharge_m3_s, {control}")
        elif mode in ("inlet_normal","inlet_critical"):
            self.series_label.setText("One row per knot: time_s, total_discharge_m3_s. The section-derived water level is resolved at each knot.")
        elif mode=="total_discharge":
            self.series_label.setText("One row per knot: time_s, total_discharge_m3_s. The water level is free.")
        else:self.series_label.setText("One row per knot: time_s, water_level_m")
        self.series_label.setVisible(series);self.series.setVisible(series)
        if self.kind=="inlet":self.plot_button.setText("Plot Time Series" if series else "Plot Forcing")
        if mode=="total_discharge":self.summary.setText("The discharge is total for this ID; the inlet water level is free. The selected profile distributes flow over the section.")
        elif mode=="discharge":self.summary.setText("Legacy characteristic discharge uses the configured water level. The accepted inlet level may evolve away from it.")
        elif mode=="inlet_normal":self.summary.setText("The discharge is total for this ID. Normal depth is computed automatically from the DEM and Mesh settings.")
        elif inlet:self.summary.setText("The discharge is total for this ID. The selected section supplies one horizontal target water line.")
        elif mode=="stage":self.summary.setText("The selected faces use the prescribed water level; local reversal remains fail-fast.")
        elif mode=="critical":self.summary.setText("Critical depth is recomputed from the current outward section discharge; reversal is rejected.")
        elif mode=="weir":self.summary.setText("Free-outflow rating: Q = coefficient × width × max(upstream level − crest, 0)^1.5. Drowning and reversal are rejected.")
        else:self.summary.setText("Information is released through the selected faces only while flow remains supercritical.")
        self._on_change()

    def set_values(self,values):
        widgets=(self.mode,self.forcing,self.profile,self.discharge,self.stage,self.depth,self.weir_coefficient,self.weir_width,self.series)
        for widget in widgets:widget.blockSignals(True)
        mode=values.get("mode","inlet_stage" if self.kind=="inlet" else "stage")
        index=self.mode.findData(mode)
        if index<0:raise ValueError(f"{self.bc_id}: boundary mode {mode!r} is not editable in this UI")
        self.mode.setCurrentIndex(index)
        profile=values.get("total_discharge_profile","stock_conveyance")
        profile_index=self.profile.findData(profile)
        if profile_index<0:raise ValueError(f"{self.bc_id}: unknown total-discharge profile {profile!r}")
        self.profile.setCurrentIndex(profile_index)
        series=values.get("hydraulic_series")
        self.forcing.setCurrentIndex(self.forcing.findData("series" if series else "constant"))
        if series:
            controls=series.get("control_m")
            rows=[]
            for index,(time_s,discharge) in enumerate(zip(series["times_s"],series["discharge_m3_s"])):
                if mode=="stage":row=(time_s,controls[index])
                elif controls is None:row=(time_s,discharge)
                else:row=(time_s,discharge,controls[index])
                rows.append(", ".join(format(float(value),".12g") for value in row))
            self.series.setPlainText("\n".join(rows))
        self.discharge.setValue(float(values.get("discharge_m3_s",0.)))
        self.stage.setValue(float(values.get("stage_m",0.)))
        self.depth.setValue(float(values.get("depth_m",0.)))
        self._energy_slope=float(values.get("energy_slope",.001));self._boundary_manning=values.get("boundary_manning_n")
        self.weir_coefficient.setValue(float(values.get("weir_coefficient",1.7)))
        self.weir_width.setValue(float(values.get("weir_width_m") or 0.))
        for widget in widgets:widget.blockSignals(False)
        self._changed()

    def set_automatic_normal_inputs(self,slope,manning):
        self._energy_slope=float(slope);self._boundary_manning=float(manning)

    def parsed_series(self):
        mode=self.mode.currentData();columns_required=3 if mode in ("discharge","inlet_stage","inlet_depth") else 2
        rows=[]
        for number,line in enumerate(self.series.toPlainText().splitlines(),1):
            text=line.strip()
            if not text or text.startswith("#"):continue
            columns=text.replace(";",",").split(",") if "," in text or ";" in text else text.split()
            if len(columns)!=columns_required:
                raise ValueError(f"{self.bc_id}: hydraulic row {number} needs exactly {columns_required} values")
            try:rows.append(tuple(float(value.strip()) for value in columns))
            except ValueError:
                if not rows and any(character.isalpha() for character in text):continue
                raise ValueError(f"{self.bc_id}: hydraulic row {number} is not numeric")
        if len(rows)<2:raise ValueError(f"{self.bc_id}: hydraulic time series requires at least two numeric rows")
        if mode=="stage":return CutHydraulicBoundarySeries(tuple(row[0] for row in rows),(0.,)*len(rows),tuple(row[1] for row in rows))
        controls=None if columns_required==2 else tuple(row[2] for row in rows)
        return CutHydraulicBoundarySeries(tuple(row[0] for row in rows),tuple(row[1] for row in rows),controls)

    def values(self):
        mode=self.mode.currentData()
        series=self.parsed_series() if self.forcing.currentData()=="series" else None
        return {"mode":mode,"discharge_m3_s":self.discharge.value() if self.kind=="inlet" and series is None else 0.,
                "stage_m":self.stage.value() if mode in ("discharge","inlet_stage","stage","weir") and series is None else 0.,
                "depth_m":self.depth.value() if series is None else 0.,"energy_slope":self._energy_slope,
                "boundary_manning_n":self._boundary_manning,
                "total_discharge_profile":self.profile.currentData() if mode=="total_discharge" else "stock_conveyance",
                "weir_coefficient":self.weir_coefficient.value(),"weir_width_m":self.weir_width.value() or None,
                "reversal_policy":"reject","hydraulic_series":None if series is None else asdict(series)}


class SedimentBoundaryCard(AccordionSection):
    """Constant or piecewise-linear per-fraction supply for one inlet ID."""
    def __init__(self,bc_id,values,on_change,on_plot,*,fraction_ids=(),expanded=False,parent=None):
        super().__init__(f"Inlet sediment · {bc_id}",expanded=expanded,parent=parent)
        self.bc_id=bc_id;self._on_change=on_change;self.fraction_ids=tuple(fraction_ids)
        form=QFormLayout();self.body_layout.addLayout(form)
        self.forcing=QComboBox();self.forcing.addItem("Constant","constant");self.forcing.addItem("Time series","series")
        self.concentration=_spin(0.,1e9,0.,6,.1)
        self.bed_basis=QComboBox();self.bed_basis.addItem("Total per boundary ID (kg/s)","total");self.bed_basis.addItem("Per metre of boundary (kg/(m·s))","per_width");self.bed_basis.addItem("Equilibrium capacity from local flow","equilibrium_capacity")
        if self.fraction_ids:
            self.bed_basis.clear();self.bed_basis.addItem("Totals listed per fraction (kg/s)","total")
            self.bed_basis.addItem("Equilibrium capacity for each fraction","equilibrium_capacity")
        self.bed_value=_spin(0.,1e12,0.,6,.1)
        form.addRow("Forcing",self.forcing);form.addRow("Concentration (kg/m³)",self.concentration);form.addRow("Bed-load input basis",self.bed_basis);form.addRow("Bed-load input",self.bed_value)
        self.series_label=QLabel("One row per knot: time_s, concentration_kg_m3, total_bed_input_kg_s")
        self.series_label.setWordWrap(True);self.series=QPlainTextEdit();self.series.setMinimumHeight(105)
        self.body_layout.addWidget(self.series_label);self.body_layout.addWidget(self.series)
        self.fraction_label=QLabel("One row per fraction: ID, concentration_kg_m3, total_bed_input_kg_s")
        self.fraction_label.setWordWrap(True);self.fraction_editor=QPlainTextEdit();self.fraction_editor.setMinimumHeight(90)
        self.body_layout.addWidget(self.fraction_label);self.body_layout.addWidget(self.fraction_editor)
        buttons=QHBoxLayout();self.plot_button=QPushButton("Plot Time Series");buttons.addStretch();buttons.addWidget(self.plot_button);self.body_layout.addLayout(buttons)
        self.plot_button.clicked.connect(lambda:on_plot(self));self.set_values(values)
        self.forcing.currentIndexChanged.connect(self._changed);self.concentration.valueChanged.connect(self._changed)
        self.bed_basis.currentIndexChanged.connect(self._changed);self.bed_value.valueChanged.connect(self._changed);self.series.textChanged.connect(self._changed)
        self.fraction_editor.textChanged.connect(self._changed)

    def _changed(self,*_):
        if self.fraction_ids:
            self.forcing.setEnabled(True);series=self.forcing.currentData()=="series"
            for widget in (self.concentration,self.bed_value):
                widget.setVisible(False);label=self.body_layout.itemAt(0).layout().labelForField(widget)
                if label is not None:label.setVisible(False)
            self.series_label.setVisible(False);self.series.setVisible(False)
            self.fraction_label.setVisible(True);self.fraction_editor.setVisible(True)
            equilibrium=not series and self.bed_basis.currentData()=="equilibrium_capacity"
            self.bed_basis.setEnabled(not series)
            self.fraction_editor.setEnabled(not equilibrium)
            self.fraction_label.setText(("One row per knot: fraction ID, time_s, concentration_kg_m3, total_bed_input_kg_s"
                                         if series else
                                         "One row per fraction: ID, concentration_kg_m3, total_bed_input_kg_s"))
            self.plot_button.setText("Plot Fraction Time Series" if series else "Plot Fraction Supply")
            self._on_change();return
        series=self.forcing.currentData()=="series"
        equilibrium=not series and self.bed_basis.currentData()=="equilibrium_capacity"
        for widget in (self.concentration,self.bed_basis):widget.setEnabled(not series)
        self.bed_value.setEnabled(not series and not equilibrium)
        self.series_label.setVisible(series);self.series.setVisible(series)
        self.fraction_label.setVisible(False);self.fraction_editor.setVisible(False)
        self.plot_button.setText("Plot Time Series" if series else ("Plot Concentration" if equilibrium else "Plot Constant Supply"))
        self._on_change()

    def set_values(self,values):
        for widget in (self.forcing,self.concentration,self.bed_basis,self.bed_value,self.series,self.fraction_editor):widget.blockSignals(True)
        if self.fraction_ids:
            concentrations=values.get("fraction_concentration_kg_m3",{})
            bed=values.get("fraction_bed_input_kg_s",{})
            fraction_series=values.get("fraction_sediment_series",{}) or {}
            if fraction_series:
                self.forcing.setCurrentIndex(self.forcing.findData("series"))
                lines=[]
                for name in self.fraction_ids:
                    series=fraction_series.get(name)
                    if series is None:continue
                    lines.extend(f"{name}, {float(t):.12g}, {float(c):.12g}, {float(b):.12g}"
                                 for t,c,b in zip(series["times_s"],series["concentration_kg_m3"],series["bed_input_kg_s"]))
                self.fraction_editor.setPlainText("\n".join(lines))
            else:
                self.forcing.setCurrentIndex(self.forcing.findData("constant"))
                self.fraction_editor.setPlainText("\n".join(
                    f"{name}, {float(concentrations.get(name,0.)):.12g}, {float(bed.get(name,0.)):.12g}"
                    for name in self.fraction_ids))
            basis="equilibrium_capacity" if not fraction_series and values.get("bedload_boundary_mode")=="equilibrium_capacity" else "total"
            self.bed_basis.setCurrentIndex(self.bed_basis.findData(basis))
        else:
            series=values.get("sediment_series")
            if series:
                self.forcing.setCurrentIndex(self.forcing.findData("series"))
                rows=zip(series["times_s"],series["concentration_kg_m3"],series["bed_input_kg_s"])
                self.series.setPlainText("\n".join(", ".join(format(float(value),".12g") for value in row) for row in rows))
            else:
                self.forcing.setCurrentIndex(self.forcing.findData("constant"));self.concentration.setValue(float(values.get("concentration_kg_m3",0.)))
                total=values.get("bed_input_kg_s")
                basis=("equilibrium_capacity" if values.get("bedload_boundary_mode")=="equilibrium_capacity" else
                       ("total" if total is not None else "per_width"))
                self.bed_basis.setCurrentIndex(self.bed_basis.findData(basis))
                self.bed_value.setValue(float(total if total is not None else values.get("bed_input_kg_m_s",0.)))
        for widget in (self.forcing,self.concentration,self.bed_basis,self.bed_value,self.series,self.fraction_editor):widget.blockSignals(False)
        self._changed()

    def parsed_fraction_supply(self):
        values={}
        for number,line in enumerate(self.fraction_editor.toPlainText().splitlines(),1):
            text=line.strip()
            if not text or text.startswith("#"):continue
            columns=text.replace(";",",").split(",") if "," in text or ";" in text else text.split()
            if len(columns)!=3:raise ValueError(f"{self.bc_id}: fraction row {number} needs ID, concentration and total bed input")
            name=columns[0].strip()
            try:values[name]=(float(columns[1]),float(columns[2]))
            except ValueError as exc:raise ValueError(f"{self.bc_id}: fraction row {number} is not numeric") from exc
        if set(values)!=set(self.fraction_ids):raise ValueError(f"{self.bc_id}: fraction supply must list every configured fraction exactly once")
        if any(min(row)<0 for row in values.values()):raise ValueError(f"{self.bc_id}: fraction supplies must be nonnegative")
        return values

    def parsed_fraction_series(self):
        grouped={name:[] for name in self.fraction_ids}
        for number,line in enumerate(self.fraction_editor.toPlainText().splitlines(),1):
            text=line.strip()
            if not text or text.startswith("#"):continue
            columns=text.replace(";",",").split(",") if "," in text or ";" in text else text.split()
            if len(columns)!=4:raise ValueError(f"{self.bc_id}: fraction-series row {number} needs ID, time, concentration and total bed input")
            name=columns[0].strip()
            if name not in grouped:raise ValueError(f"{self.bc_id}: fraction-series row {number} names unknown fraction {name!r}")
            try:grouped[name].append(tuple(float(value) for value in columns[1:]))
            except ValueError as exc:raise ValueError(f"{self.bc_id}: fraction-series row {number} is not numeric") from exc
        missing=[name for name,rows in grouped.items() if len(rows)<2]
        if missing:raise ValueError(f"{self.bc_id}: every fraction needs at least two time-series rows: {', '.join(missing)}")
        return {name:CutSedimentBoundarySeries(tuple(row[0] for row in rows),tuple(row[1] for row in rows),
                                               tuple(row[2] for row in rows)) for name,rows in grouped.items()}

    def parsed_series(self):
        rows=[]
        for number,line in enumerate(self.series.toPlainText().splitlines(),1):
            text=line.strip()
            if not text or text.startswith("#"):continue
            columns=text.replace(";",",").split(",") if "," in text or ";" in text else text.split()
            if len(columns)!=3:
                raise ValueError(f"{self.bc_id}: time-series row {number} needs exactly three values")
            try:rows.append(tuple(float(value.strip()) for value in columns))
            except ValueError:
                if not rows and any(character.isalpha() for character in text):continue
                raise ValueError(f"{self.bc_id}: time-series row {number} is not numeric")
        if len(rows)<2:raise ValueError(f"{self.bc_id}: time series requires at least two numeric rows")
        series=CutSedimentBoundarySeries(tuple(row[0] for row in rows),tuple(row[1] for row in rows),tuple(row[2] for row in rows))
        return series

    def values(self):
        if self.fraction_ids:
            if self.forcing.currentData()=="series":
                return {"concentration_kg_m3":0.,"bed_input_kg_m_s":0.,"bed_input_kg_s":None,
                        "bedload_boundary_mode":"prescribed","sediment_series":None,
                        "fraction_concentration_kg_m3":{},"fraction_bed_input_kg_s":{},
                        "fraction_sediment_series":{name:asdict(series) for name,series in self.parsed_fraction_series().items()}}
            equilibrium=self.bed_basis.currentData()=="equilibrium_capacity"
            supplies=self.parsed_fraction_supply()
            return {"concentration_kg_m3":0.,"bed_input_kg_m_s":0.,"bed_input_kg_s":None,
                    "bedload_boundary_mode":"equilibrium_capacity" if equilibrium else "prescribed",
                    "sediment_series":None,
                    "fraction_concentration_kg_m3":{name:row[0] for name,row in supplies.items()},
                    "fraction_bed_input_kg_s":{} if equilibrium else {name:row[1] for name,row in supplies.items()},
                    "fraction_sediment_series":{}}
        if self.forcing.currentData()=="series":
            return {"concentration_kg_m3":0.,"bed_input_kg_m_s":0.,"bed_input_kg_s":None,
                    "bedload_boundary_mode":"prescribed","sediment_series":asdict(self.parsed_series())}
        if self.bed_basis.currentData()=="equilibrium_capacity":
            return {"concentration_kg_m3":self.concentration.value(),"bed_input_kg_m_s":0.,
                    "bed_input_kg_s":None,"bedload_boundary_mode":"equilibrium_capacity","sediment_series":None}
        total=self.bed_basis.currentData()=="total"
        return {"concentration_kg_m3":self.concentration.value(),
                "bed_input_kg_m_s":0. if total else self.bed_value.value(),
                "bed_input_kg_s":self.bed_value.value() if total else None,
                "bedload_boundary_mode":"prescribed","sediment_series":None}
