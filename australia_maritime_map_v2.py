#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, textwrap, urllib.request, zipfile
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod, CRS, Transformer
from shapely.geometry import LineString, box
from shapely.ops import unary_union, transform as shp_transform, linemerge

GA_EEZ=('https://services.ga.gov.au/gis/rest/services/SSLA_1973/MapServer/8/query'
        '?where=1%3D1&outFields=*&returnGeometry=true&f=geojson&outSR=4326')
GA_IDN=('https://services.ga.gov.au/gis/rest/services/Treaties_Australian_Maritime_Boundaries/MapServer/22/query'
        '?where=1%3D1&outFields=*&returnGeometry=true&f=geojson&outSR=4326')
MARINE='https://raw.githubusercontent.com/lsdch/countries-boundaries/main/data/EEZ_land_union_v4_202410.json'
MAP='https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_map_units.zip'
LAND='https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip'
EXPECTED={'Indonesia','Papua New Guinea','Timor-Leste','Solomon Islands','New Zealand','France'}
DISPLAY={'France':'France (New Caledonia / Kerguelen)'}
LOCATORS=[('Cocos (Keeling) Islands',96.83,-12.17),('Christmas Island',105.63,-10.49),('Ashmore and Cartier Islands',123.08,-12.26),('Coral Sea Islands Territory',149.97,-16.29),('Norfolk Island',167.95,-29.03),('Heard Island and McDonald Islands',73.50,-53.10),('Lord Howe Island (NSW)',159.08,-31.55),('Macquarie Island (Tas.)',158.94,-54.62)]
WATER='#cfeaf6'; LANDC='#d6d6d6'; COAST='#777'; RED='#c62828'; GREY='#737373'; TEXT='#202020'; PANEL='#e7f4fa'
GEOD=Geod(ellps='WGS84')
MAPCRS=CRS.from_proj4('+proj=aea +lat_1=-15 +lat_2=-45 +lat_0=-30 +lon_0=132 +datum=WGS84 +units=m +no_defs')

def dl(url,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size>1000:return
    req=urllib.request.Request(url,headers={'User-Agent':'Australia maritime map v2'})
    with urllib.request.urlopen(req,timeout=240) as r,p.open('wb') as f:
        while True:
            b=r.read(1024*1024)
            if not b:break
            f.write(b)

def lines(g):
    if g is None or g.is_empty:return []
    if g.geom_type in ('LineString','LinearRing'):return [LineString(g.coords)]
    if g.geom_type in ('MultiLineString','GeometryCollection'):return [z for x in g.geoms for z in lines(x)]
    if g.geom_type in ('Polygon','MultiPolygon'):return lines(g.boundary)
    return []
def polys(g):
    if g is None or g.is_empty:return []
    if g.geom_type=='Polygon':return [g]
    if g.geom_type in ('MultiPolygon','GeometryCollection'):return [z for x in g.geoms for z in polys(x)]
    return []
def glen(g):
    s=0.0
    for l in lines(g):
        a=np.asarray(l.coords,float)
        if len(a)>1:s+=float(np.abs(GEOD.inv(a[:-1,0],a[:-1,1],a[1:,0],a[1:,1])[2]).sum())
    return s
def merge(g):
    try:return linemerge(unary_union(lines(g)))
    except:return unary_union(lines(g))
def remove_ant(g):
    keep=[]
    for p in polys(g):
        if p.representative_point().y>-59.75:keep += polys(p.intersection(box(-180,-59.75,180,90)))
    return unary_union(keep)
def col(g,names):
    d={str(c).lower():c for c in g.columns}
    for n in names:
        if n.lower() in d:return d[n.lower()]
    return None
def textmask(g,pat):
    m=pd.Series(False,index=g.index)
    for c in g.columns:
        if c!=g.geometry.name and (g[c].dtype==object or str(g[c].dtype).startswith('string')):
            m|=g[c].fillna('').astype(str).str.contains(pat,case=False,regex=True)
    return m
def cname(v,f=''):
    t=(str(v)+' '+str(f)).lower()
    if 'papua new guinea' in t:return 'Papua New Guinea'
    if 'timor' in t:return 'Timor-Leste'
    if 'solomon' in t:return 'Solomon Islands'
    if 'new zealand' in t:return 'New Zealand'
    if 'new caledonia' in t or 'france' in t or 'french' in t or 'kerguelen' in t:return 'France'
    if 'indonesia' in t:return 'Indonesia'
    return ''
def extract(z,d):
    d.mkdir(parents=True,exist_ok=True)
    if not list(d.rglob('*.shp')):
        with zipfile.ZipFile(z) as q:q.extractall(d)
    return next(d.rglob('*.shp'))

def labels_on_borders(ax,projected):
    for country,g in projected.items():
        segs=sorted(lines(merge(g)),key=lambda s:s.length,reverse=True)
        keep=[s for s in segs if s.length>45000]
        if not keep and segs:keep=[segs[0]]
        for i,s in enumerate(keep[:5]):
            p=s.interpolate(.5,normalized=True)
            txt=ax.annotate('France' if country=='France' else country,(p.x,p.y),
                xytext=(0,10 if i%2==0 else -12),textcoords='offset points',ha='center',va='center',
                fontsize=8.2,fontweight='bold',color='#8e1111',zorder=20,
                bbox=dict(boxstyle='round,pad=.22',fc=WATER,ec='none',alpha=.92))
            txt.set_path_effects([pe.withStroke(linewidth=2.2,foreground=WATER),pe.Normal()])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path('output'));ap.add_argument('--data-dir',type=Path,default=Path('data'));a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);a.data_dir.mkdir(parents=True,exist_ok=True)
    files={'ga':a.data_dir/'ga_eez.geojson','idn':a.data_dir/'ga_idn.geojson','marine':a.data_dir/'marine.json','map':a.data_dir/'map.zip','land':a.data_dir/'land.zip'}
    for u,k in [(GA_EEZ,'ga'),(GA_IDN,'idn'),(MARINE,'marine'),(MAP,'map'),(LAND,'land')]:print('Downloading',u);dl(u,files[k])
    # Official Australian Government EEZ standard depiction.
    eez=gpd.read_file(files['ga']).to_crs(4326);aus=remove_ant(unary_union(eez.geometry));outer=aus.boundary
    to=Transformer.from_crs(4326,3857,always_xy=True).transform;fr=Transformer.from_crs(3857,4326,always_xy=True).transform;outerm=shp_transform(to,outer)
    shared={}
    # Indonesia directly from the official GA 1997 EEZ treaty-line service. This is
    # essential for the Christmas Island–Java segment, which is absent in the third-party topology.
    idn=gpd.read_file(files['idn']).to_crs(4326);idnline=unary_union(idn.geometry)
    idnnear=shp_transform(fr,outerm.intersection(shp_transform(to,idnline).buffer(250)))
    shared['Indonesia']=merge(idnnear)
    # Other neighbour names from Marine Regions, clipped to the official Australian outer boundary.
    world=gpd.read_file(files['marine']).to_crs(4326);world=world[world.geometry.notna() & ~world.geometry.is_empty].copy()
    sov=col(world,['SOVEREIGN1','SOVEREIGN','SOVEREIGNT','ADMIN']);ter=col(world,['TERRITORY1','TERRITORY','GEONAME','NAME']);typ=col(world,['POL_TYPE','POLTYPE','TYPE']);iso=col(world,['ISO_SOV1','SOV_A3','ADM0_A3','ISO3','ISO_A3'])
    am=pd.Series(False,index=world.index)
    if sov:am|=world[sov].fillna('').astype(str).str.fullmatch(r'\s*Australia\s*',case=False)
    if iso:am|=world[iso].fillna('').astype(str).str.upper().isin(['AUS','AU1'])
    amb=pd.Series(False,index=world.index)
    if typ:amb=world[typ].fillna('').astype(str).str.contains(r'joint|overlap|disput|claim',case=False,regex=True)
    foreign=world.loc[~am & ~amb & ~textmask(world,'Antarct')]
    for c in sorted(EXPECTED-{'Indonesia'}):
        bs=[r.geometry.boundary for _,r in foreign.iterrows() if cname(r.get(sov,'') if sov else '',r.get(ter,'') if ter else '')==c]
        if not bs:continue
        near=shp_transform(fr,outerm.intersection(shp_transform(to,unary_union(bs)).buffer(150)))
        if glen(near)>1000:shared[c]=merge(near)
    if set(shared)!=EXPECTED:raise RuntimeError(f'Neighbour QA failed: {sorted(shared)}')
    # Explicit Christmas Island QA: the standard Australia–Indonesia line must occur northwest/north of Christmas Island.
    christmas=shared['Indonesia'].intersection(box(102,-14.2,108.5,-8))
    if glen(christmas)<10000:raise RuntimeError(f'Christmas Island Indonesia boundary QA failed: {glen(christmas)/1000:.2f} km')
    su=unary_union(list(shared.values()));high=outer.difference(su);lengths={c:glen(g)/1000 for c,g in shared.items()}
    # Land depiction.
    us=extract(files['map'],a.data_dir/'mapshp');ps=extract(files['land'],a.data_dir/'landshp');units=gpd.read_file(us).to_crs(4326);phys=gpd.read_file(ps).to_crs(4326)
    usov=col(units,['SOVEREIGN','SOVEREIGNT','ADMIN']);uiso=col(units,['SOV_A3','ADM0_A3','ISO_A3']);um=pd.Series(False,index=units.index)
    if usov:um|=units[usov].fillna('').astype(str).str.contains('Australia',case=False)
    if uiso:um|=units[uiso].fillna('').astype(str).str.upper().isin(['AUS','AU1'])
    um &= ~textmask(units,'Antarct');landparts=list(units.loc[um].geometry)
    regions=unary_union([box(96.6,-12.45,97.2,-11.7),box(105.4,-10.75,105.9,-10.25),box(122.7,-12.75,123.75,-11.65),box(149.7,-16.6,150.2,-16),box(167.7,-29.3,168.15,-28.75),box(72.3,-53.45,74.05,-52.75),box(158.7,-54.85,159.2,-54.35),box(158.9,-31.8,159.3,-31.3)])
    for g in phys.geometry:
        if g is not None and not g.is_empty and g.intersects(regions):landparts.append(g.intersection(regions))
    land=remove_ant(unary_union(landparts))
    tr=Transformer.from_crs(4326,MAPCRS,always_xy=True);proj=lambda g:shp_transform(tr.transform,g)
    lp,apg,hp=proj(land),proj(aus),proj(high);sp={c:proj(g) for c,g in shared.items()}
    fig=plt.figure(figsize=(18,10.3),facecolor=WATER);gs=fig.add_gridspec(1,2,width_ratios=[4.9,1.65],wspace=0);ax=fig.add_subplot(gs[0,0]);panel=fig.add_subplot(gs[0,1]);ax.set_facecolor(WATER);panel.set_facecolor(WATER);ax.axis('off');panel.axis('off')
    minx,miny,maxx,maxy=apg.bounds;w=maxx-minx;h=maxy-miny;ax.set_xlim(minx-.025*w,maxx+.025*w);ax.set_ylim(miny-.035*h,maxy+.035*h);ax.set_aspect('equal')
    for p in polys(lp):x,y=p.exterior.xy;ax.fill(x,y,facecolor=LANDC,edgecolor=COAST,lw=.38,zorder=3)
    for l in lines(hp):x,y=l.xy;ax.plot(x,y,color=GREY,lw=1.05,alpha=.42,zorder=4)
    for g in sp.values():
        for l in lines(g):x,y=l.xy;ax.plot(x,y,color=RED,lw=2.1,alpha=.98,zorder=6)
    labels_on_borders(ax,sp)
    for label,lon,lat in LOCATORS:
        x,y=tr.transform(lon,lat);ax.scatter([x],[y],s=16,facecolors=LANDC,edgecolors=COAST,lw=.75,zorder=9);ax.annotate(label,(x,y),xytext=(8,8),textcoords='offset points',fontsize=7,color=TEXT,zorder=9)
    ax.text(.015,.985,"AUSTRALIA’S MARITIME BOUNDARIES",transform=ax.transAxes,ha='left',va='top',fontsize=19,fontweight='bold',color=TEXT)
    ax.text(.016,.948,'EEZ / water-column outer limit • all Australian territories shown except Antarctica',transform=ax.transAxes,ha='left',va='top',fontsize=9.8,color='#4f4f4f')
    card=FancyBboxPatch((.045,.06),.90,.88,boxstyle='round,pad=.018,rounding_size=.025',lw=.8,edgecolor='#9bbdca',facecolor=PANEL,transform=panel.transAxes);panel.add_patch(card)
    panel.text(.10,.895,'BOUNDARY KEY',transform=panel.transAxes,fontsize=10.5,fontweight='bold');panel.plot([.10,.23],[.842,.842],transform=panel.transAxes,color=RED,lw=2.4);panel.text(.27,.842,'Shared / delimitation with country',transform=panel.transAxes,fontsize=8.8,va='center');panel.plot([.10,.23],[.796,.796],transform=panel.transAxes,color=GREY,lw=1.4,alpha=.45);panel.text(.27,.796,'Outer limit adjoining high seas',transform=panel.transAxes,fontsize=8.8,va='center')
    panel.text(.10,.70,'COUNTRIES BY TOTAL SHARED LENGTH',transform=panel.transAxes,fontsize=10.5,fontweight='bold');panel.text(.10,.67,'WGS 84 ellipsoidal distance • nearest kilometre',transform=panel.transAxes,fontsize=7.4,color='#555')
    y=.615
    for rank,(c,km) in enumerate(sorted(lengths.items(),key=lambda q:-q[1]),1):
        panel.text(.10,y,str(rank),transform=panel.transAxes,fontsize=9.4,fontweight='bold',color=RED);panel.text(.16,y,DISPLAY.get(c,c),transform=panel.transAxes,fontsize=8.8);panel.text(.89,y,f'{km:,.0f} km',transform=panel.transAxes,fontsize=9.2,ha='right',family='DejaVu Sans Mono');y-=.06
    panel.text(.10,.205,'INDONESIA / CHRISTMAS ISLAND',transform=panel.transAxes,fontsize=9.2,fontweight='bold')
    note="Christmas Island’s Indonesia-facing boundary is included. Australia’s EEZ uses Geoscience Australia’s standard Perth-Treaty-adjusted SSLA depiction. The 1997 Australia–Indonesia treaty is signed but not yet in force; GA specifies the adjusted depiction for standard operational use."
    panel.text(.10,.177,textwrap.fill(note,48),transform=panel.transAxes,fontsize=7.0,color='#4c4c4c',va='top',linespacing=1.3)
    fig.text(.012,.012,'Australian EEZ + Indonesia line: Geoscience Australia. Other neighbour classification: Marine Regions/VLIZ v4. Land: Natural Earth 1:10m. Lengths: WGS 84. Antarctica excluded.',fontsize=6.6,color='#555')
    fig.savefig(a.output_dir/'australia_maritime_boundaries.png',dpi=300,bbox_inches='tight',facecolor=WATER);fig.savefig(a.output_dir/'australia_maritime_boundaries.svg',bbox_inches='tight',facecolor=WATER);plt.close(fig)
    qa={'lengths_km':dict(sorted(lengths.items(),key=lambda q:-q[1])),'christmas_island_indonesia_boundary_km':glen(christmas)/1000,'neighbours':sorted(shared)}
    (a.output_dir/'qa.json').write_text(json.dumps(qa,indent=2));print('PASS',json.dumps(qa))
if __name__=='__main__':main()
