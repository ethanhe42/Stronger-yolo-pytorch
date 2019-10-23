from trainers.base_trainer import BaseTrainer
from evaluator.cocoeval import EvaluatorCOCO

class Trainer(BaseTrainer):
  def __init__(self, args, model, optimizer,lrscheduler):
    super().__init__(args, model, optimizer,lrscheduler)

  def _get_loggers(self):
    #get public loggers
    super()._get_loggers()
    self.TESTevaluator = EvaluatorCOCO(anchors=None,
                                       cateNames=self.labels,
                                       rootpath=self.dataset_root,
                                       score_thres=0.01,
                                       iou_thres=0.5,
                                       )
    # get customized loggers
    self.logger_custom = ['mAP', 'mAp@50', 'mAP@75', 'mAP@small', 'mAP@meduim', 'mAP@large',
                        'AR@1', 'AR@10', 'AR@100', 'AR@small', 'AR@medium', 'AR@large']


