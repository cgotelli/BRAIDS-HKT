# pyvista_postprocess.py
from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors
from clawpack.geoclaw import fgout_tools
from argparse import ArgumentParser



#installer cmocean "pip install cmocean" pour avoir cmo.topo

pv.global_theme.allow_empty_mesh = True

def animation(outdir    = "_output",
              color_var = "dh",
              fgno      = 1,
              cmaps     = ("gist_earth", "RdBu"), #"cmo.topo"
              clim      = (-3, 3),
              bathy_clim= None,
              file_name = "animation.mp4",
              init_frame= 1,
              animate   = False,
              world_png = "",
              colorbar_label = "",
              window_size    = (1600, 800),
              keep_images    = False,
              out_video      = None,
              frames_per_s   = 12,
              azimuth        = 0,
              elevation      = -60):
    current_directory = Path.cwd()
    if out_video is not None:
        if isinstance(out_video,str):
            video_dir = current_directory.parent / 'Figures' / out_video
            # si out_video n'est pas une chaîne de caractères, on utilise le dossier par défaut
        else:
            video_dir = current_directory.parent / 'Figures' / "Vidéos"
    else:
        # si out_video = None, alors on utilise le dossier par défaut
        video_dir = current_directory.parent / 'Figures' / "Vidéos"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"I use {video_dir} for saving the video (if any)")
    if keep_images:
        animation_dir = current_directory.parent / 'Figures' / "Animation" / "Lake"
        print(f"I use {animation_dir} for keeping the exported images (if any)")
        animation_dir.mkdir(parents=True, exist_ok=True)
    outdir     = Path(outdir)
    fgout_grid = fgout_tools.FGoutGrid(fgno, outdir)
    fgout_grid.read_fgout_grids_data()

    X = fgout_grid.X
    Y = fgout_grid.Y

    fgout_init = fgout_grid.read_frame(init_frame)
    fgout_init.dh = fgout_init.eta - fgout_init.eta

    if bathy_clim is None:
        b = fgout_init.B
        # Centre la colormap sur le niveau du lac (surface libre des cellules mouillées)
        # pour que la transition eau/terre de cmo.topo tombe au bon endroit
        wet = fgout_init.h > 0
        if wet.any():
            lake_level = float(fgout_init.eta[wet].mean())
        else:
            lake_level = 0.0
        bathy_max = float(b.max())
        bathy_min = float(b.min())
        # Pour cmo.topo, la transition est au milieu de la plage : on symétrise autour de lake_level
        #athy_clim = (2 * lake_level - bathy_max, bathy_max)
        bathy_clim = (bathy_min, bathy_max)
    p = pv.Plotter()

    # Paramètres de la barre scalaire
    # PyVista infère l'orientation depuis le ratio : width > height → horizontale
    # position_x/position_y = coin inférieur gauche (fraction de la fenêtre)
    sargs = dict(

    width           = 0.5,
    height          = 0.1,
    position_x      = 0.05,
    position_y      = 0.05,
    title_font_size = 24,
    label_font_size = 20,
    shadow          = True,
    n_labels        = 3,
    italic          = False,
    fmt             = '{0:.1f}',
    font_family     = 'times',
    title           = r'$\Delta h$ (m)',   # PyVista/VTK ne supporte pas LaTeX natif
    )

    bathy = pv.StructuredGrid(X, Y, fgout_init.B)
    if world_png:
        tex = pv.read_texture(world_png)
        tex.repeat = False
        nx, ny, nc = tex.to_image().dimensions
        
        dx, r1, r2, dy, xmin, ymax = np.loadtxt(Path(world_png).with_suffix(".pgw"))
        xmax = xmin + nx * dx
        ymin = ymax + ny * dy
        o = xmin, ymin, 0
        u = xmax, ymin, 0
        v = xmin, ymax, 0
        bathy.texture_map_to_plane(origin=o, point_u=u, point_v=v, inplace=True)
        scalars=None
    else:
        # Hillshade calculé depuis B, mêmes paramètres que plot_topo de module_avac
        # X a la forme (nx, ny) → B.T donne (ny, nx) = (lignes=y, colonnes=x) pour hillshade()
        ls  = mcolors.LightSource(azdeg=315, altdeg=45)
        dx  = float(X[1, 0] - X[0, 0])
        hs  = ls.hillshade(fgout_init.B.T, vert_exag=3, dx=dx, dy=dx)  # (ny, nx), valeurs [0,1]
        # PyVista/VTK place l'origine de la texture en bas → numpy (origine en haut) doit être retourné
        hs_rgb = (np.stack([np.flipud(hs)]*3, axis=-1) * 255).astype(np.uint8)
        tex = pv.Texture(hs_rgb)
        xmin_b, xmax_b = float(X.min()), float(X.max())
        ymin_b, ymax_b = float(Y.min()), float(Y.max())
        bathy.texture_map_to_plane(
            origin=(xmin_b, ymin_b, 0),
            point_u=(xmax_b, ymin_b, 0),
            point_v=(xmin_b, ymax_b, 0),
            inplace=True
        )
        scalars = None
    p.add_mesh(bathy, texture=tex, scalars=scalars, cmap=cmaps[0], clim=bathy_clim, show_scalar_bar=False)

    surf = pv.StructuredGrid(X, Y, np.where(fgout_init.h>0, fgout_init.eta, np.nan))
    # B = fgout_init.B
    # B = np.flipud(B)
    # surf.point_data['B'] = B.flatten(order='F')
    # warpfactor = 2 # amplification of elevations
    #topowarp = surf.warp_by_scalar('B', factor=warpfactor)

    label       = colorbar_label or color_var
    surf[label] = get_value(fgout_init, fgout_init, color_var).T.flatten()
    #topowarp[label] = get_value(fgout_init, fgout_init, color_var).T.flatten()
    p.add_mesh(surf, scalars=label, cmap=cmaps[1], clim=clim, show_scalar_bar=True,scalar_bar_args=sargs)
    #p.add_mesh(topowarp, scalars=label, cmap=cmaps[1], clim=clim, show_scalar_bar=True)

    # Texte du temps en coordonnées pixels (retourne un vtkTextActor avec SetInput)
    time_actor = p.add_text(
        rf"$t = {fgout_grid.times[init_frame-1]:.1f}\,\mathrm{{s}}$",
        position    = (1000, 60),
        font_size   = 14,
        color       = 'black',
        font        = 'times'
    )
    state = dict(i=init_frame)
    def update(i):
        fgout = fgout_grid.read_frame(i)
        fgout.dh = fgout.eta - fgout_init.eta
        bathy.points[:, 2] = fgout.B.T.flatten()
        surf.points[:, 2] = np.where(fgout.h>0, fgout.eta, np.nan).T.flatten()
        surf[label] = get_value(fgout, fgout_init, color_var).T.flatten()
        time_actor.SetInput(rf"$t = {fgout_grid.times[i-1]:.1f}\,\mathrm{{s}}$")
        #topowarp.points[:, 2] = np.where(fgout.h>0, fgout.eta, np.nan).T.flatten()
        #topowarp[label] = get_value(fgout, fgout_init, color_var).T.flatten()

    def update_index(increment=None, value=None):
        if increment is not None:
            state["i"] += increment
        else:
            state["i"] = value
        update(state["i"])
        p.update()

    def prev_frame():
        update_index(increment=-1)
    def next_frame():
        update_index(increment=+1)
    def prevv_frame():
        update_index(increment=-10)
    def nextt_frame():
        update_index(increment=+10)

    def save_movie():
        import subprocess
        import tempfile
        fname = file_name or "animation.mp4"
        fname = str(video_dir / fname)
        s = state["i"]
        print(f"Enregistrement de {fname} (frame {s} → fin)...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for n, i in enumerate(range(s, len(fgout_grid.times) + 1)):
                update(i)
                p.render()
                p.screenshot(str(tmpdir / f"lake_{n:05d}.png"))
            if keep_images:
                import shutil
                shutil.copytree(tmpdir, animation_dir, dirs_exist_ok=True)

            if Path(fname).suffix.lower() == ".gif":
                subprocess.run([
                    "ffmpeg", "-y",
                    "-framerate",str(frames_per_s),
                    "-i", str(tmpdir / "lake_%05d.png"),
                    "-vf", "fps=12,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                    fname
                ], check=True)
            else:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-framerate", str(frames_per_s),
                    "-i", str(tmpdir / "lake_%05d.png"),
                    "-vcodec",  "libx264",
                    "-pix_fmt", "yuv420p",
                    fname
                ], check=True)
        print(f"Sauvegardé : {fname}")
    
    def save_image():
        snapshot = "snapshot_"
        i = max([int(path.stem[len(snapshot):]) for path in Path().glob(f"{snapshot}*.svg")] or [0]) + 1
        filename = f"{snapshot}{i}.svg"
        p.save_graphic(
            filename,
            title=f"Frame {state['i']}: t={fgout_grid.times[state['i']-1]}"
        )

    # p.view_xy()          # vue de dessus (caméra au-dessus, regardant vers le bas)
    # p.camera.azimuth = 0  # rotation dans le plan horizontal (0 = Nord en haut)
    p.camera_position  = 'xy'   # caméra au-dessus
    p.camera.elevation = elevation   # 60° sous l'horizontale  
    p.camera.azimuth   = azimuth


    if animate is True:
        p.add_timer_event(max_steps=len(fgout_grid.times), duration=500, callback=update)
        p.show(window_size=window_size)

    else:
        p.add_key_event("h", prevv_frame)
        p.add_key_event("j", prev_frame)
        p.add_key_event("k", next_frame)
        p.add_key_event("l", nextt_frame)
        p.add_key_event("m", save_movie)
        p.add_key_event("s", save_image)
        p.add_key_event("Left", prev_frame)
        p.add_key_event("Right", next_frame)
        p.show(window_size=window_size)
        

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--outdir",         "-o", type=str, default="_output")
    parser.add_argument("--color_var",      "-c", type=str, default="dh")
    parser.add_argument("--fgno",           "-n", type=int, default=1)
    parser.add_argument("--cmaps",          "-m", type=str, nargs=2, default=("Greys", "hsv"))
    parser.add_argument("--clim",           "-l", type=float, nargs=2, default=(-0.5, 0.5))
    parser.add_argument("--file_name",      "-f", type=str, default="animation_vagues.mp4")
    parser.add_argument("--init_frame",     "-i", type=int, default=1)
    parser.add_argument("--animate",        "-a", action="store_true")
    parser.add_argument("--world_png",      "-w", type=Path)
    parser.add_argument("--colorbar_label", "-t", type=str)
    parser.add_argument("--window_size",    "-ws", type=int, nargs=2, default=(1600,1000))
    parser.add_argument("--keep_images",    "-ki", type=bool,default = False)
    parser.add_argument("--out_video",      "-ov", type=str, default = None)
    parser.add_argument("--frames_per_s",   "-fps", type=int,default = 12)
    parser.add_argument("--azimuth",        "-az", type=float,default = 0)
    parser.add_argument("--elevation",      "-el", type=float,default = -60)
    args = parser.parse_args()
    return args

def get_value(fgout, fgout0, name):
    if name[0] == "d" and hasattr(fgout, name[1:]):
        return getattr(fgout, name[1:]) - getattr(fgout0, name[1:])
    return getattr(fgout, name)

 

def main():
    args = parse_args()
    animation(**args.__dict__)

if __name__ == "__main__":
    main()