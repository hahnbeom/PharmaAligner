import os,glob

# 클러스터 그룹 종류 받아오기(CB,CC,...)
os.system("find ~knw423/tarmol2s.gaff -maxdepth 1 -type d > clusters_name.txt")
clusters = [line.rstrip() for line in open('clusters_name.txt')]
clusters.pop(0)
clusters.sort()

for I in range(len(clusters)) : # 각 그룹별(CB,CG..)로 .tar 파일들 이름 리스트 받아옴
    print(clusters[I])
    path1 = 'npzs/%s' %clusters[I][-3::1]
    os.makedirs(path1, exist_ok=True)
    os.system("find %s | grep '.tar' > temp.txt" %clusters[I])#.tar파일들 이름 temp.txt에 저장
    fpath1 = 'temp.txt'

    lines = [line.rstrip() for line in open('temp.txt')]
    lines.sort()
    lines.pop(0)
    #print(lines)

    for tarf in lines:#CB.0, CB.1 이런식으로 탐색
        path = tarf.split('/')[-1][:-4]
        os.system('tar -xvf %s 1>/dev/null'%tarf)
        os.chdir(path)

        mol2s = glob.glob('*mol2')
        name = path.split('_')[1]+'.'+path.split('_')[-1]+'.mol2'#CB.0.mol2 이런식으로 mol2파일 생성
        feats = {}
        
        os.system("scp ../conformers1.mol2 %s"%path)

        f = open(name,'a+')
        lst = []
        for m in mol2s:
            print(m)
            words = m[:-5].split('_')
            prefix = '.'.join([words[k] for k in [1,-2,-1]])
            os.system("obabel %s -O conformers1.mol2 --conformer --nconf 30 --writeconformers"%m)
            os.system("python ../usr.py conformers1.mol2")
            
            new = open('new_conformers.mol2','r')#new_conformers.mol2:usr.py에서 최종적으로 탐색할 후보들 mol2로 만든 것
            f.writelines(new.readlines())#CB.0.mol2파일에 CB.0.1, CB.0.2 등에 해당하는 conformers 붙여 넣어줌
            new.close()

            cnt = 0
            for l in open("new_conformers.mol2"):
                if l.startswith('@<TRIPOS>MOLECULE'):
                    cnt += 1
                    data = "%s.conf%d\n" %(prefix ,cnt)#conformer별로 이름 다르게 저장해야, npz만들 수 있으므로
                    lst.append(data)

        with open('name_list.txt','w') as file:
            file.writelines(lst)#CB.0.1.conf1,CB.0.1.conf2,CB.0.2.conf1 이런식의 이름이 저장된 txt 파일

        f.close()
        
        os.chdir('..')
        os.system("~hpark/anaconda3/bin/python featurize_input.py %s/%s %s/name_list.txt" %(path,name,path))#npz로 저장시, 파일들 이름 필요하므로, name_list.txt 이용함. 결국 넣어주는 것은 new_conformers.mol2, name_list.txt
        os.system("rm -rf %s" %path)

