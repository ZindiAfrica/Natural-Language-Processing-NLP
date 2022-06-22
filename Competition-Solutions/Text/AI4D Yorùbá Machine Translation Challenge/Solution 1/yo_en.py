from tqdm import tqdm
import mtranslate as translator
import pandas as pd
class Translate_yo_en():
  def from_pandas(self,df_path):
    self.df = df_path
    self.col = pd.read_csv(self.df)["input_text"]
    self.col = self.col.dropna()
    preds = []
    for i in tqdm(self.col):
      preds.append(translator.translate(i,"en","yo"))
    return preds
  def from_string(self,string):
    self.string = string
    return translator.translate(self.string,"en","yo")