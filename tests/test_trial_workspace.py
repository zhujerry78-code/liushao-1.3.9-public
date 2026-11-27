from contextlib import closing

import pandas as pd

from app_input import input_times, preflight_input
from app_quality import QualityView, KNOWN_REJECTION
from app_local_workspace import (clear_reviews, download_history, last_position,
    log_download, recent, save_position)
from app_session import clear_upload, consent_generation, saved_upload, save_upload
from app_types import ActiveInput
from liushao import recovery, reviews


def test_time_coverage_and_zone_basis():
    values = pd.Series(['02/03/2018 01:02:17', '2018-03-02T03:02:17+02:00',
                        '2018-03-02 01:02:17', 'invalid', ''])
    stamps, basis = input_times(values)
    assert stamps.iloc[0] == stamps.iloc[1] == stamps.iloc[2]
    assert basis.iloc[1] == 'UTC' and stamps.isna().sum() == 2
    content = ('x,Timestamp\n'+'\n'.join('1,'+v for v in values)+'\n').encode()
    preflight = preflight_input(content, {'config':{'features':['x']}}, 'model')
    assert preflight.time_counts == (3,1,1)


def test_quality_details_show_adapter_reasons():
    index = pd.Index(['sample'])
    fields = ['empty','text','inf','range']
    view = QualityView(pd.DataFrame(),pd.DataFrame({'reason':[KNOWN_REJECTION],
        'disposition':['无法分析']},index=index),4,1,1,0,1,0,1,
        pd.DataFrame([['','oops','-inf','1e99']],columns=fields,index=index),
        pd.DataFrame([[True]*4],columns=fields,index=index))
    assert view.details('sample')['适配结果'].tolist()==['缺值','非数字','非有限数值','超出模型数值范围']


def test_positions_and_scoped_cleanup(tmp_path):
    database = tmp_path / 'reviews.sqlite'
    active = ActiveInput(b'x','input.csv','上传 CSV','a'*64,'key','model')
    reviews.save_review(database,'record','model','待补证','自动化测试依据')
    save_position(database,active,{'record':'record','filters':{},'pending':1})
    log_download(database,active,'test.csv','rev1','选定事件')
    assert last_position(database,active)['record']=='record' and len(recent(database))==1
    save_upload(database,b'x','input.csv','model',consent_generation(database))
    clear_upload(database)
    assert saved_upload(database) is None and len(reviews.history(database))==1
    assert len(download_history(database,active))==1
    save_upload(database,b'x','input.csv','model',consent_generation(database))
    with closing(reviews.connect(database)) as db:
        db.execute('ATTACH DATABASE ? AS recovery',(str(database)+'.recovery.sqlite',))
        recovery.initialize_recovery(db)
        counts=clear_reviews(db,database)
    assert counts['main.reviews']==1
    assert reviews.history(database)==[] and recent(database)==[] and download_history(database,active)==[]
    assert saved_upload(database)['name']=='input.csv'

