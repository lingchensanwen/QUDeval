import openai
import os
import openai
import pandas as pd
import os
from openai import OpenAI
import re
import numpy as np
from sklearn.metrics import f1_score


ANCHOR_FILE_PATH = "../../data/example_anchor_answer_info.csv"
VAL_ESSAY_PATH = "../../data/example_article.txt"
PRED_LABEL_PATH = "tune_info_labels.csv"
BEST_THRESHOLD = "best_threshold_info.txt"

def extract_number_from_string(input_string):
  number = re.findall(r'\d+', input_string)
  if number:
      return int(number[0])
  else:
      print("this is not a number")
  return "empty"



def get_few_shots_chatgpt_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-4-0613", #"gpt-3.5-turbo" for chatgpt, "gpt-4-0613" for gpt4
    messages = [
        {"role": "system", "content": '''
        article: U.S. exports of nuclear material cannot be adequately traced from country to country, according to a congressional report.'Scarcely a day goes by without a report of a new black market deal,' said Sen. John Glenn in a statement reacting to the report.'Given the staggering amount of nuclear materials we have exported, it could only be a matter of time before some of this deadly contraband proves to be of U.S.origin.'As chairman of the Senate Committee on Governmental Affairs in the last Congress, Glenn commissioned the report from the General Accounting Office, which conducts investigations for legislators.The report says hundreds of tons of plutonium and highly enriched uranium have accumulated worldwide, mostly from nuclear power generation.It does not include figures on U.S. nuclear exports but says 71 export licenses for nuclear materials were granted in 1993.Nuclear exports for weapons use or weapons research are prohibited, as is transfer of nuclear materials to a third country.The report said U.S. tracking showed that Japan produced about 20.3 metric tons of plutonium from U.S. materials between 1978 and 1992, but Japanese records showed 58.7 tons.A nuclear weapon can be made with as little as 6 kilograms (13.2 pounds) of plutonium, U.S. officials have said.'We're looking at those kinds of discrepancies and we are confident that the material is being safeguarded as appropriate by the IAEA (International Atomic Energy Agency,)' said Notra Trulock, chief of intelligence at the U.S. Department of Energy, in an interview Tuesday.The report quoted a U.S. official working on reconciliation of the two figures as saying one reason for the discrepancy was that Japan was only required to report to the United States the plutonium it sent to other countries for reprocessing.Trulock said he did not want to comment on U.S. nuclear material sent to Europe because he was engaged in delicate negotiations on renewing the U.S. agreement with EURATOM, the nuclear section of the European Union.The congressional report noted that the 12 countries in EURATOM -- 15 countries since Jan. 1 -- have been treated as a unit for purposes of reporting transfers of nuclear material.'The U.S. agreement with .(EURATOM) does not require most EURATOM countries to inform the United States of retransfers of U.S.-supplied materials from one EURATOM country to another.,' the report complained.Trulock said the GAO had interviewed him for the report.It said officials of the Energy Department, the State Department and other agencies generally agreed with the facts it presented.It added that they had offered the explanation that the size and complexity of the government's old tracking system was such that the Department of Energy had not yet improved it, but only adapted it for personal computers.The statement said the system should have planned and designed properly to begin with.
        question: What do foreign policy experts say about the issue?
        answer: 'Scarcely a day goes by without a report of a new black market deal,' said Sen. John Glenn in a statement reacting to the report. 
        Score:0
        
        
        article:Amid skepticism that Russia's war in Chechnya can be ended across a negotiating table, peace talks were set to resume Wednesday in neighboring Ingushetia.The scheduled resumption of talks in the town of Sleptsovsk came two days after agreement on a limited cease-fire, calling for both sides to stop using heavy artillery Tuesday.They also agreed in principle to work out a mechanism for exchanging prisoners of war and the dead.Despite the pact, artillery fire sounded in the Grozny on Tuesday, and there were reports of Chechen missile attacks southwest of the Chechen capital.Many Chechens are fighting independently of the forces loyal to Chechen President Dzhokhar Dudayev, and Dudayev's representative at the peace talks, Aslan Maskhadov, has warned that he does not control them.Russian Prime Minister Viktor Chernomyrdin said Tuesday that Moscow is also willing to talk to individual field commanders, the Interfax news agency reported.Similar offers in the past have been ignored by the feisty Chechens.The talks Wednesday were to be held in Sleptsovsk with the Russian side represented by Col. Gen. Anatoly Kulikov, commander of Russian troops in Chechnya.Many doubt the cease-fire can hold given the bitterness of the conflict.Emil Pain, an adviser to President Boris Yeltsin, was quoted as saying a 'cooling period after combat operations' is needed before negotiations could take place.Thousands of people have been killed and more than 400,000 displaced since Russian forces stormed into Chechnya on Dec. 11 to crush Dudayev's government and end the region's self-proclaimed independence.The Russian offensive has turned Grozny into a wasteland littered with rotting bodies, twisted metal and debris.Hardly a building is untouched.The war has also cost Russia dearly -- in lives, prestige and rubles.In Stockholm, both Britain and Sweden pressed Russia's foreign minister Tuesday to explain Moscow's goals in Chechnya.The questions came during a flurry of meetings between British Foreign Secretary Douglas Hurd, Russian Foreign Minister Andrei Kozyrev and their Swedish hosts.Chechnya 'is a cloud over reform and our support for reform,' Hurd said before the meeting.Many experts say the expense of subduing Chechnya could bust the budget and wreck Russia's attempt to refashion its economy along free market lines.The head of a special commission assigned to supervise Chechnya's reconstruction said Tuesday the cost could be three times the Russian government's estimate.Arkady Volsky told Associated Press Television that the political and economic instability the war has caused has driven the overall cost -- including lost foreign and domestic investment -- to dlrs 40 billion.
        question:What does the U.S. government expect from the Japanese regarding the issue?
        answer:Thousands of people have been killed and more than 400,000 displaced since Russian forces stormed into Chechnya on Dec. 11 to crush Dudayev's government and end the region's self-proclaimed independence.
        Score:100
         
    
        article:FORT LAUDERDALE, Fla. - Researchers are looking to the sun to give hunted and overfished sharks a new ray of hope.Using a special solar-powered tag, marine scientists now can study a shark's movements for up to two years by way of data beamed to satellites.Previously, researchers relied on tags that ran on batteries and sometimes died before all the information could be transmitted.The new tags are like 'a smartphone for marine animals,' said Marco Flagg, CEO of Desert Star, a Marina, Calif., company that offers the solar devices.'Just like smartphones, the tags have many sensors and communication capability'.The Guy Harvey Research Institute, based in Dania Beach, Fla., is looking to use the solar tags to track certain species of the fierce fish, including tigers, makos, hammerheads, oceanic white tip and sand sharks.The goal is to better understand their migratory patterns and ultimately keep their population healthy.Sharks are critical to the overall balance of ocean ecosystems, but commercial fishermen catch them by the millions for their fins, cartilage and meat.'We've learned a lot from tagging sharks, not least of which is that they are highly migratory,' said Antonio Fins, executive director of the Guy Harvey Ocean Foundation, which supports the institute.'They are not American sharks or Bahamian sharks or Mexican sharks.They don't know borders or nationalities'.About 40 research agencies already use solar tags, which were put on the market two years ago.For instance, the University of Miami's Rosenstiel School of Marine &amp; Atmospheric Sciences studies a variety of sharks, while others use them to track turtles and marine mammals that spend time in the sun.The overall success of solar tags has yet to be proven because of their relatively limited use.But so far marine researchers have encountered no serious problems, and a growing number of agencies plan to purchase them, manufactures said.By drawing on solar energy, the tags ensure power is available to beam to a satellite a range of data, including how deep the fish go and the water temperatures they encounter.That information is then transmitted to researchers.Because most sharks don't linger near the surface - in direct sunlight - the solar-powered tags are programmed to collect data for about six months while running on conventional batteries.Then the tags detach and float to the surface, said Mahmood Shivji, director of the Guy Harvey Research Institute, part of Nova Southeastern University.'Now it's exposed to sunlight,' Shivji said, 'and it's been archiving data for six months'.
        question:Why are researchers studying sharks and using solar-powered tags to track their movements?
        answer:Sharks are critical to the overall balance of ocean ecosystems, but commercial fishermen catch them by the millions for their fins, cartilage and meat.
        Score:50

        For the above article, give a score between 1 to 100 for how well the answer actually answers the question.
         
        '''},
        {"role":"user", "content": prompt}],
    temperature=0, #0
    max_tokens=128
  )

    result = response.choices[0].message.content
    return result

## helper methods
def retrieve_article_by_id(data_df, article_id):
  for i in range(len(data_df)):
    if(data_df.iloc[i][0][0]['ArticleID'] == article_id):
      return data_df.iloc[i][0][0]
    
## this is for random sampling, we sample once and read from it everytime as we do not 
## have that many annotators to annotate all qud for all sentences from one article and do this for all.
## in this example article we just take 1-10 as samples but generally we're doing random sample
def read_sampling_index(df):
    sample_file_path = "../../data/example_sampling.csv"
    sampled_df = pd.read_csv(sample_file_path)
    sampled_indices = list(sampled_df['selected_row_number'] - 1)
    sampled_df = df.loc[sampled_indices].sort_index()

    return sampled_df

def read_context(ESSAY_PATH):
    with open(ESSAY_PATH, 'r') as f:
        essay_lines = f.readlines()
    essay_context = [line.strip().split("\t")[1] for i, line in enumerate(essay_lines)]
    return essay_context


def generate_predict_labels(essay_path):
    with open(BEST_THRESHOLD, 'r') as file:
        line = file.readline().strip()

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    best_thresh1, best_thresh2 = map(float, numbers[:2])
    print(f"best threshold 1 and 2 read is {best_thresh1} and {best_thresh2}")

    df = pd.read_csv(ANCHOR_FILE_PATH)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "index"})

    df = read_sampling_index(df)
    questions = []
    context = read_context(essay_path)
    response = []
    for _, row in df.iterrows():
        question_text = str(row['questions']).replace('<|endoftext|>', '')
        answer_id = int(row["answer_id"])
        answer_sentence = context[answer_id-1]
        prompt = "article: " + ''.join(context) + "\n"+ "question: " + question_text + "\n"+ "answer: " + answer_sentence + "\n"
        result = get_few_shots_chatgpt_response(prompt)
        score = extract_number_from_string(result)
        if score > best_thresh2:
            response.append(1)
        elif score > best_thresh1:
            response.append(2)
        else:
            response.append(3)

    response_df = pd.DataFrame({
        "response": response,
        "anchor_id": df["anchor_id"],
        "answer_id": df["answer_id"],
        "questions": df["questions"].replace('', '')
    })    

    response_df.to_csv(PRED_LABEL_PATH, index=False)
    print("result saved to " + PRED_LABEL_PATH)

    

def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = os.getenv('OPENAI_ORGANIZATION')
    generate_predict_labels(VAL_ESSAY_PATH)

if __name__ == "__main__":
    main()

