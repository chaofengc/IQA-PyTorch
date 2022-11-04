- [[IQA toolbox]]
  - 
    - `python  model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img']) model.feed_data(train_data) model.optimize_parameters(current_iter)`
  - bapps_model.
    - [[how to write metric]]
    - general iqa model is the base_model
    - it use [[pixel]] loss
    - why here is exp
      - `python def optimize_parameters(self, current_iter):                self.optimizer.zero_grad()                score_A = self.net(self.img_A_input, self.img_ref_input)        score_B = self.net(self.img_B_input, self.img_ref_input)        # For BAPPS,         train_output_score = (1 / (1 + torch.exp(score_B - score_A)))                l_total = 0        loss_dict = OrderedDict()        # pixel loss        if self.cri_mos:            l_mos = self.cri_mos(train_output_score, self.gt_mos)            l_total += l_mos            loss_dict['l_mos'] = l_mos         l_total.backward()        self.optimizer.step()         self.log_dict = self.reduce_loss_dict(loss_dict)         # log metrics in training batch                self.log_dict[f'train_metrics/acc'] = self.compute_accuracy(score_A, score_B, self.gt_mos) `
    - [[diffence between test and train ,eval]]
      - `python    def test(self):        self.net.eval()        with torch.no_grad():            self.score_A = self.net(self.img_A_input, self.img_ref_input)            self.score_B = self.net(self.img_B_input, self.img_ref_input)        self.net.train()`
    - ![img](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FfinalYearProject%2F4_r1G5wXv4.png?alt=media&token=607410b7-af17-41d2-a849-cd6fe0ad3a06)
  - the data composition `python        self.img_A_input = data['distA_img'].to(self.device)        self.img_B_input = data['distB_img'].to(self.device)        self.img_ref_input = data['ref_img'].to(self.device)        self.gt_mos = data['mos_label'].to(self.device)        self.img_path = data['img_path']`
  - baseModel
    - what is dist(distribution)
  - what is time embedding:
    - seems to be a boolean value
  - what is linear attention

 what is data made of

SRModel(BaseModel):

"""Base SR model for single image super-resolution."""

class BAPPSModel(GeneralIQAModel):

  """General module to train an IQA network."""

class GeneralIqaModel