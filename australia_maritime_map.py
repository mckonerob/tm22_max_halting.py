#!/usr/bin/env python3
import argparse, json, re, textwrap, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod, CRS, Transformer
from shapely.geometry import GeometryCollection, LineString, Polygon, box
from shapely.ops import unary_union, transform as shp_transform

MARINE_URL='https://raw.githubusercontent.com/lsdch/countries-boundaries/main/data/EEZ_land_union_v4_202410.json'
MAP_URL='https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_map_units.zip'
LAND_URL='https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip'
EXPECTED={'Indonesia','Papua New Guinea','Timor-Leste','Solomon Islands','New Zealand','France'}
DISPLAY={'France':'France (New Caledonia)'}
LOCATORS=[('Cocos (Keeling) Islands',96.83,-12.17),('Christmas Island',105.63,-10.49),('Ashmore and Cartier Islands',123.08,-12.26),('Coral Sea Islands Territory',149.97,-16.29),('Norfolk Island',167.95,-29.03),('Heard Island and McDonald Islands',73.50,-53.10),('Lord Howe Island (NSW)',159.08,-31.55),('Macquarie Island (Tas.)',158.94,-54.62)]
WATER='#cfeaf6'; LAND='#d6d6d6'; COAST='#777777'; RED='#c62828'; GREY='#737373'; TEXT='#202020'; PANEL='#e7f4fa'
GEOD=Geod(ellps='WGS84')
CRS_MAP=CRS.from_proj4('+proj=aea +lat_1=-15 +lat_2=-45 +lat_0=-30 +lon_0=132 +datum=WGS84 +units=m +no_defs')

def dl(url,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.stat().st_size>1000:return
 req=urllib.request.Request(url,headers={'User-Agent':'Australia maritime map/1.0'})
 with urllib.request.urlopen(req,timeout=240) as r, open(p,'wb') as f:
  while True:
   b=r.read(1024*1024)
   if not b:break
   f.write(b)

def parts(g):
 if g is None or g.is_empty:return []
 if g.geom_type in ('LineString','LinearRing'):return [LineString(g.coords)]
 if g.geom_type in ('MultiLineString','GeometryCollection'):return [x for q in g.geoms for x in parts(q)]
 if g.geom_type in ('Polygon','MultiPolygon'):return parts(g.boundary)
 return []
def polys(g):
 if g is None or g.is_empty:return []
 if g.geom_type=='Polygon':return [g]
 if g.geom_type in ('MultiPolygon','GeometryCollection'):return [x for q in g.geoms for x in polys(q)]
 return []
def length(g):
 s=0
 for l in parts(g):
  a=np.asarray(l.coords,float)
  if len(a)>1:s+=float(np.abs(GEOD.inv(a[:-1,0],a[:-1,1],a[1:,0],a[1:,1])[2]).sum())
 return s
def country(v,f=''):
 t=(str(v)+' '+str(f)).lower()
 for k,p in [('Indonesia','indonesia'),('Papua New Guinea','papua new guinea'),('Timor-Leste','timor'),('Solomon Islands','solomon'),('New Zealand','new zealand'),('France','new caledonia')]:
  if p in t:return k
 if 'france' in t or 'french' in t:return 'France'
 return str(v).strip()
def col(g,names):
 d={str(x).lower():x for x in g.columns}
 for n in names:
  if n.lower() in d:return d[n.lower()]
 return None
def txtmask(g,pat):
 m=pd.Series(False,index=g.index)
 for c in g.columns:
  if c!=g.geometry.name and (g[c].dtype==object or str(g[c].dtype).startswith('string')):m|=g[c].fillna('').astype(str).str.contains(pat,case=False,regex=True)
 return m
def extract(z,d):
 d.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(z) as q:q.extractall(d)
 return next(d.rglob('*.shp'))
def remove_ant(g):
 keep=[]
 for p in polys(g):
  if p.representative_point().y>-59.75:
   keep += polys(p.intersection(box(-180,-59.75,180,90)))
 return unary_union(keep)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=Path('output')); ap.add_argument('--data-dir',type=Path,default=Path('data')); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); a.data_dir.mkdir(parents=True,exist_ok=True)
 marine=a.data_dir/'EEZ_land_union_v4_202410.json'; mz=a.data_dir/'map.zip'; lz=a.data_dir/'land.zip'
 for u,p in [(MARINE_URL,marine),(MAP_URL,mz),(LAND_URL,lz)]: print('Downloading',u); dl(u,p)
 world=gpd.read_file(marine).to_crs(4326); world=world[world.geometry.notna() & ~world.geometry.is_empty].copy()
 sov=col(world,['SOVEREIGN1','SOVEREIGN','SOVEREIGNT','ADMIN']); iso=col(world,['ISO_SOV1','SOV_A3','ADM0_A3','ISO3','ISO_A3']); typ=col(world,['POL_TYPE','POLTYPE','TYPE']); ter=col(world,['TERRITORY1','TERRITORY','GEONAME','NAME'])
 am=pd.Series(False,index=world.index)
 if sov:am|=world[sov].fillna('').astype(str).str.fullmatch(r'\s*Australia\s*',case=False)
 if iso:am|=world[iso].fillna('').astype(str).str.upper().isin(['AUS','AU1'])
 if not am.any():am=txtmask(world,r'\bAustralia\b')
 amb=pd.Series(False,index=world.index)
 if typ:amb=world[typ].fillna('').astype(str).str.contains(r'joint|overlap|disput|claim',case=False,regex=True)
 ant=txtmask(world,'Antarct')
 ag=remove_ant(unary_union(world.loc[am & ~amb & ~ant].geometry)); foreign=world.loc[~am & ~amb & ~ant]
 ab=ag.boundary; shared=defaultdict(list)
 for _,r in foreign.iterrows():
  c=country(r.get(sov,'') if sov else '',r.get(ter,'') if ter else '')
  if c not in EXPECTED:continue
  try:i=ab.intersection(r.geometry.boundary)
  except:continue
  if length(i)>1000:shared[c].append(i)
 shared={c:unary_union(v) for c,v in shared.items()}
 missing=EXPECTED-set(shared)
 if missing:
  # Repair tiny topology offsets only for expected neighbours.
  to=Transformer.from_crs(4326,3857,always_xy=True).transform; fr=Transformer.from_crs(3857,4326,always_xy=True).transform; aw=shp_transform(to,ab)
  for c in list(missing):
   bs=[]
   for _,r in foreign.iterrows():
    if country(r.get(sov,'') if sov else '',r.get(ter,'') if ter else '')==c:bs.append(r.geometry.boundary)
   if bs:
    near=shp_transform(fr,aw.intersection(shp_transform(to,unary_union(bs)).buffer(50)))
    if length(near)>1000:shared[c]=near
 if set(shared)!=EXPECTED:raise RuntimeError('Neighbour validation failed: '+repr(sorted(set(shared))))
 su=unary_union(list(shared.values())); high=ab.difference(su); lengths={c:length(g)/1000 for c,g in shared.items()}
 # Land: Natural Earth map units plus physical land around small Australian territories.
 us=extract(mz,a.data_dir/'map'); ps=extract(lz,a.data_dir/'land')
 units=gpd.read_file(us).to_crs(4326); phys=gpd.read_file(ps).to_crs(4326); usov=col(units,['SOVEREIGN','SOVEREIGNT','ADMIN']); uiso=col(units,['SOV_A3','ADM0_A3','ISO_A3'])
 um=pd.Series(False,index=units.index)
 if usov:um|=units[usov].fillna('').astype(str).str.contains('Australia',case=False)
 if uiso:um|=units[uiso].fillna('').astype(str).str.upper().isin(['AUS','AU1'])
 if not um.any():um=txtmask(units,r'\bAustralia\b')
 um &= ~txtmask(units,'Antarct'); landparts=list(units.loc[um].geometry)
 boxes=[box(96.6,-12.45,97.2,-11.7),box(105.4,-10.75,105.9,-10.25),box(122.7,-12.75,123.75,-11.65),box(149.7,-16.6,150.2,-16),box(155.3,-23.5,155.8,-23),box(167.7,-29.3,168.15,-28.75),box(72.3,-53.45,74.05,-52.75),box(158.7,-54.85,159.2,-54.35),box(158.9,-31.8,159.3,-31.3)]
 reg=unary_union(boxes)
 for g in phys.geometry:
  if g is not None and not g.is_empty and g.intersects(reg):landparts.append(g.intersection(reg))
 land=remove_ant(unary_union(landparts))
 tr=Transformer.from_crs(4326,CRS_MAP,always_xy=True); proj=lambda g:shp_transform(tr.transform,g)
 lp,apg,hp=proj(land),proj(ag),proj(high); sp={c:proj(g) for c,g in shared.items()}
 fig=plt.figure(figsize=(18,10.3),facecolor=WATER); gs=fig.add_gridspec(1,2,width_ratios=[4.9,1.65],wspace=0); ax=fig.add_subplot(gs[0,0]); panel=fig.add_subplot(gs[0,1]); ax.set_facecolor(WATER);panel.set_facecolor(WATER);ax.axis('off');panel.axis('off')
 minx,miny,maxx,maxy=apg.bounds; w=maxx-minx;h=maxy-miny;ax.set_xlim(minx-.025*w,maxx+.025*w);ax.set_ylim(miny-.035*h,maxy+.035*h);ax.set_aspect('equal')
 for p in polys(lp):
  x,y=p.exterior.xy;ax.fill(x,y,facecolor=LAND,edgecolor=COAST,linewidth=.38,zorder=3)
 for l in parts(hp):x,y=l.xy;ax.plot(x,y,color=GREY,lw=1.05,alpha=.42,zorder=4)
 for g in sp.values():
  for l in parts(g):x,y=l.xy;ax.plot(x,y,color=RED,lw=2.05,alpha=.96,zorder=6)
 for label,lon,lat in LOCATORS:
  x,y=tr.transform(lon,lat);ax.scatter([x],[y],s=16,facecolors=LAND,edgecolors=COAST,lw=.75,zorder=7);ax.annotate(label,(x,y),xytext=(8,8),textcoords='offset points',fontsize=7,color=TEXT,zorder=8)
 ax.text(.015,.985,"AUSTRALIA’S MARITIME BOUNDARIES",transform=ax.transAxes,ha='left',va='top',fontsize=19,fontweight='bold',color=TEXT);ax.text(.016,.948,'EEZ / water-column outer limit • all Australian territories shown except Antarctica',transform=ax.transAxes,ha='left',va='top',fontsize=9.8,color='#4f4f4f')
 card=FancyBboxPatch((.045,.06),.90,.88,boxstyle='round,pad=.018,rounding_size=.025',lw=.8,edgecolor='#9bbdca',facecolor=PANEL,transform=panel.transAxes);panel.add_patch(card);panel.text(.10,.895,'BOUNDARY KEY',transform=panel.transAxes,fontsize=10.5,fontweight='bold',color=TEXT);panel.plot([.10,.23],[.842,.842],transform=panel.transAxes,color=RED,lw=2.4);panel.text(.27,.842,'Shared with another country',transform=panel.transAxes,fontsize=8.8,va='center');panel.plot([.10,.23],[.796,.796],transform=panel.transAxes,color=GREY,lw=1.4,alpha=.45);panel.text(.27,.796,'Outer limit adjoining high seas',transform=panel.transAxes,fontsize=8.8,va='center')
 panel.text(.10,.70,'COUNTRIES BY TOTAL SHARED LENGTH',transform=panel.transAxes,fontsize=10.5,fontweight='bold');panel.text(.10,.67,'WGS 84 ellipsoidal distance • nearest kilometre',transform=panel.transAxes,fontsize=7.4,color='#555')
 y=.615
 for rank,(c,km) in enumerate(sorted(lengths.items(),key=lambda q:-q[1]),1):panel.text(.10,y,str(rank),transform=panel.transAxes,fontsize=9.4,fontweight='bold',color=RED);panel.text(.16,y,DISPLAY.get(c,c),transform=panel.transAxes,fontsize=9.2);panel.text(.89,y,f'{km:,.0f} km',transform=panel.transAxes,fontsize=9.2,ha='right',family='DejaVu Sans Mono');y-=.06
 panel.text(.10,.205,'SCOPE',transform=panel.transAxes,fontsize=9.6,fontweight='bold');scope="Red and grey partition Australia's EEZ / water-column outer limit. Seabed and continental-shelf lines are not mixed into this map. Australian Antarctic Territory is excluded.";panel.text(.10,.177,textwrap.fill(scope,48),transform=panel.transAxes,fontsize=7.3,color='#4c4c4c',va='top',linespacing=1.35)
 fig.text(.012,.012,'Geometry: Marine Regions / VLIZ Union EEZ–land v4 (Oct 2024; DOI 10.14284/698). Land: Natural Earth 1:10m. Lengths: WGS 84 geodesic. Antarctica excluded.',fontsize=6.9,color='#555')
 png=a.output_dir/'australia_maritime_boundaries.png';svg=a.output_dir/'australia_maritime_boundaries.svg';fig.savefig(png,dpi=300,bbox_inches='tight',facecolor=WATER);fig.savefig(svg,bbox_inches='tight',facecolor=WATER);plt.close(fig)
 (a.output_dir/'lengths.json').write_text(json.dumps(dict(sorted(lengths.items(),key=lambda q:-q[1])),indent=2));print('PASS',lengths)
if __name__=='__main__':main()
