import os

all_sites = [x[1] for x in os.walk('./scPDB/')][0]

f = open('./pro_emb/listCavTagged', 'w')
for i, site in enumerate(all_sites):
    os.system('./FuzCav/utils/CaTagger.pl site.mol2 > ./pro_emb/site${i}_Tagged.mol2') # generate tagged site
    f.write(f'site{i}_Tagged.mol2' + '\n')
f.colse()

os.system('java -jar ../FuzCav/dist/3pointPharCav.jar -d ../FuzCav/utils/resDef/tableDefCA.txt -t ../FuzCav/utils/triplCav/interval.txt -l ./listCavTagged -o ./pro_fp.txt -c') # generate fingerprint


