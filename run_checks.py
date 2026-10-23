"""Execute declared local checks in a disposable source copy; never publish."""
import hashlib,json,os,pathlib,shutil,subprocess,sys,tempfile,time
ROOT=pathlib.Path(__file__).resolve().parent
if __name__=='__main__':
    spec=json.loads((ROOT/'RELEASE_SPEC.json').read_text())
    results=[]
    with tempfile.TemporaryDirectory(prefix='invariant-check-') as td:
        work=pathlib.Path(td)/'source'
        shutil.copytree(ROOT,work,ignore=shutil.ignore_patterns('__pycache__','.git','.venv','node_modules','checks','dist'))
        home=pathlib.Path(td)/'home';home.mkdir()
        env=dict(os.environ,HOME=str(home),MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
        env['PYTHONPATH']=os.pathsep.join([str(work/spec['check_cwd']/'02_ENGINEERING/source'),str(work/'SHARED_COMPONENTS'),str(work/spec['check_cwd']),str(work),str(work/'reference')])
        for argv in spec['commands']:
            args=[sys.executable if a=='python' else a for a in argv]
            begin=time.monotonic()
            try:
                p=subprocess.run(args,cwd=work/spec['check_cwd'],env=env,capture_output=True,text=True,timeout=300)
                results.append({'command':argv,'exit_code':p.returncode,'seconds':round(time.monotonic()-begin,3),'stdout':p.stdout,'stderr':p.stderr})
            except subprocess.TimeoutExpired as e:
                results.append({'command':argv,'exit_code':124,'seconds':300,'stdout':str(e.stdout or ''),'stderr':'TIMEOUT'})
        generated={str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in work.rglob('replay_result.json')}
        saved=[]
        for fp in work.rglob('replay_result.json'):
            target=ROOT/'checks'/'generated'/fp.relative_to(work);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fp,target);saved.append(str(target.relative_to(ROOT)))
        result={'repo' :spec['repo'],'recovery':'CP03R1','tests_executed':len(results),'passed':bool(results) and all(x['exit_code']==0 for x in results),'scope':spec['scope'],'commands':results,'generated_result_hashes':generated}
    (ROOT/'checks').mkdir(exist_ok=True)
    (ROOT/'checks/RESULT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['repo','tests_executed','passed']}))
    sys.exit(0 if result['passed'] else 1)
