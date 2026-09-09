module MakieExt

using AtmosphericTurbulenceSimulator, Makie, Unitful

# Slider label formatter: render a Unitful quantity but write inverse units with a slash
# (e.g. "50 cm/s" instead of "50 cm s⁻¹").
slashfmt(q) = replace(string(q), r" (\S+?)⁻¹" => s"/\1")

function AtmosphericTurbulenceSimulator.speckle_viewer(::Type{T}=Float64;
        wavelength_range = (550:10:750)nm,
        bw_range = (0:10:100)nm,
        exptime_range = (0:0.1:3)s,
        r0_range = (10:2:30)cm,
        wind_range = (0:5:100)cm/s,
        aperture = CircularAperture((64, 64)),
        d = 2m,
        deviceadapter=ComputeBackend()
    ) where T

    fig = Figure(size=(1200, 800))

    Label(fig[1, 1:2], "Atmosphere", font=:bold, fontsize=18, tellwidth=false)
    atm_sg = SliderGrid(fig[2, 1:2], valign=:top, alignmode=Inside(),
        (label="r₀", range=r0_range, startvalue=first(r0_range)),
        (label="Wind", range=wind_range, startvalue=first(wind_range), format=slashfmt))
    r0_s, wind_s = (s.value for s in atm_sg.sliders)
    buttons_grid = GridLayout(atm_sg.layout[3, 2], tellwidth=false, halign=:left, colgap=5)
    newphase_btn = Button(buttons_grid[1, 1], label="New phase", halign=:left)
    remove_tiptilt_btn = Button(buttons_grid[1, 2], label="Remove tip/tilt", halign=:left)

    Label(fig[1, 3:4], "Imaging", font=:bold, fontsize=18, tellwidth=false)
    img_sg = SliderGrid(fig[2, 3:4], valign=:top, alignmode=Inside(),
        (label="Wavelength", range=wavelength_range, startvalue=first(wavelength_range)),
        (label="Bandwidth", range=bw_range, startvalue=first(bw_range)),
        (label="Exposure", range=exptime_range, startvalue=first(exptime_range)))
    wl_s, bw_s, exp_s = (s.value for s in img_sg.sliders)

    phase_screen = lift(newphase_btn.clicks) do _
        atm = SingleLayer(T, first(r0_range); interpolate=:auto)
        pad = NoUnits(maximum(wind_range) / d * maximum(exptime_range))
        simulate_phases(atm, ceil.(Int, size(aperture) .* (1 + pad, 1));
            n=1, verbose=false, grid_step=d / size(aperture, 2), deviceadapter=deviceadapter)
    end

    on(remove_tiptilt_btn.clicks) do _
        phs_tot = phase_screen[]
        phs = phs_tot[axes(aperture)..., 1]
        ox, oy = size(phs) .÷ 2
        coords = hcat((axes(phs, 1) .- ox) .* ones(size(phs, 2))' |> vec,
            ones(size(phs, 1)) .* (axes(phs, 2) .- oy)' |> vec) .* vec(aperture)
        txy = coords' * vec(phs)
        gxy = coords' * coords
        cx, cy = gxy \ txy
        phase_screen[] = phs_tot .- cx .* (axes(phs_tot, 1) .- ox) .- cy .* (axes(phs_tot, 2) .- oy)'
    end

    phase_screen_scaled = lift(phase_screen, r0_s) do phs_tot, r0
        phs_tot * (first(r0_range) / r0)^(5/6)
    end

    speckle_pattern = lift(wind_s, wl_s, bw_s, exp_s, phase_screen_scaled) do wind, wl, bw, exptime, phs
        atm = SavedPhases(phs, wind_velocity=(wind, zero(wind)))
        img_spec = ImagingSpec(T, aperture, d, PhotonCount(Inf);
            filter=FilterSpec(wl; bandwidth=bw, npts=iszero(bw) ? 0 : ceil(Int, NoUnits(bw / wl) * 20 + 3)),
            exposure=Exposure(exptime))
        out = simulate_images(atm, img_spec; n=1, verbose=false, savephases=false, deviceadapter=deviceadapter)
        return Float32.(out.images[:, :, 1])
    end

    axis_kw = (aspect=DataAspect(), xticklabelspace=24.0, yticklabelspace=30.0)
    cbar_height(ax) = lift(vp -> vp.widths[2], ax.scene.viewport)
    ap_step = d / maximum(size(aperture))
    ap_x, ap_y = ustrip.(axes(aperture) .* ap_step)

    phase_screen_2d = lift(phs -> phs[axes(aperture)..., 1], phase_screen_scaled)
    ax_p, hm_p = heatmap(fig[3, 2], ap_x, ap_y, phase_screen_2d, colormap=:viridis,
        axis=(; xlabel=string(unit(d)), axis_kw...))
    contour!(ax_p, ap_x, ap_y, aperture, levels=[0.5], color=:white, linewidth=2)
    Colorbar(fig[3, 1], hm_p, label="Phase Screen (rad)", ticks=MultiplesTicks(5, pi, "π"),
        flipaxis=false, height=cbar_height(ax_p), tellheight=false, valign=:center, ticklabelspace=48.0)

    ax_i, hm_i = heatmap(fig[3, 3], speckle_pattern, colormap=:inferno, axis=axis_kw)
    Colorbar(fig[3, 4], hm_i, label="PSF (normalized)", tellheight=false,
        height=cbar_height(ax_i), valign=:center, ticklabelspace=48.0)
    hidedecorations!(ax_i)

    Label(fig[4, :], """
    Hint: drag or scroll to zoom, left-click to pan, ctrl+click to reset view.
    """, fontsize=12)
    fig
end

end # module MakieExt
