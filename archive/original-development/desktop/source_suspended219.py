"""Hash reference PDFs, final source and qualifying scripts; no vendor edits."""
from stage_suspended219 import ROOT,OUT,TARGET,REPORT,FILES,sha,save

def main():
    sources=[
      dict(file='julio.pdf',title='Zyserman and Fredsoe 1994',origin='user attachment',
           equations='Eq9-11 skin boundary layer; preferred procedure C; Eq23 concentration; p1027 Rubey settling; p1038 summary'),
      dict(file='benoit.pdf',title='Camenen and Larson 2008',origin='user attachment',
           equations='Eq4 mean transport; Eq9,11,12 diffusivity; Eq16 printed reversed threshold ratio; Eq17 grain dependence'),
      dict(file='validation/vendor/suspension/classic219/van_rijn1984.pdf',title='Van Rijn 1984 Part II',
           origin='https://www.leovanrijn-sediment.com/papers/P2-1984b.pdf',equations='11-13,19,22,33-34,37-38; printed1616-1633'),
      dict(file='validation/vendor/suspension/classic219/van_rijn1984_part1.pdf',title='Van Rijn 1984 Part I',
           origin='https://www.leovanrijn-sediment.com/papers/P2-1984a.pdf',equations='Figure1, printed1434 five-branch Shields threshold'),
      dict(file='validation/vendor/bedload/camenen_larson2005/author_2007_report.pdf',title='Camenen and Larson 2007 report',
           origin='https://erdc-library.erdc.dren.mil/server/api/core/bitstreams/81b728f8-6e74-4ef8-e053-411ac80adeb3/content',
           equations='185 supports threshold orientation; its older coefficient186 is NOT used')]
    for row in sources:row['sha256']=sha(ROOT/row['file'])
    record=dict(runtime=sha(TARGET/'RUNTIME.json'),references=sources,
        online_corroboration=[
          'https://doi.org/10.1051/e3sconf/20184004014',
          'https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/erosion-and-sediment-transport-under-construction/fall-velocity-and-settling'],
        source={n:sha(ROOT/n) for n in FILES},
        test=sha(ROOT/'tests/test_suspended_classic219.py'),
        scope='Named 2DH closures and numerical coupling verification, not full original-model experimental reproduction',
        delft3d_modified=False)
    save(OUT/'SOURCES.json',record);save(REPORT/'SOURCES.json',record)

if __name__=='__main__':main()
