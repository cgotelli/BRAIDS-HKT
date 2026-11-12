"""Authenticated isolated preview34; preserve all preview33 reference payloads."""
import stage_camenen192 as stage
ROOT=stage.ROOT
BASE=ROOT/'validation/_camenen192/candidate'
OUT=ROOT/'validation/_wong193'
TARGET=OUT/'candidate'
BASE_HASH='3f5f7954aeb95081fe66a9ca4faae404ce4e71f86b612cc764ea184163efbd81'
FILES=stage.FILES
sha,save,verify=stage.sha,stage.save,stage.verify
def configure():
    stage.BASE,stage.OUT,stage.TARGET,stage.BASE_HASH=BASE,OUT,TARGET,BASE_HASH
if __name__=='__main__':
    configure();stage.main()
