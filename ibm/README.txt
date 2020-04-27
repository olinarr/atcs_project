NAME: IBM Debater(R) - Claim Stance Dataset

VERSION: v1

RELEASE DATE: December 19, 2017

DATASET OVERVIEW 

2,394 labeled claims for 55 topics. The dataset includes the stance (Pro/Con) of each claim towards the topic, 
as well as fine-grained annotations, based on the semantic model of Bar-Haim et al. [2017a] (topic target, 
topic sentiment towards its target, claim target, claim sentiment towards its target, and the relation 
between the targets).  

The dataset includes: 
1. A utf-8 JSON file containing the topics and the claims found for these topics in Wikipedia articles. 
   Topics and claims are annotated as described above. 
2. A utf-8 CSV file containing the same information as the JSON file.
3. The original Wikipedia articles - from Wikipedia April 2012 dump - in the form of text files. 
   For each article, we provide both the original (raw) version, and a clean version, in which any Wikisyntax 
   and HTML markup is removed.
4. A CSV index file, containing the article title and Wikipedia URL for each article.   

The dataset is divided into a training set (25 topics, 1,039 claims) and a test set (30 topics, 1,355 claims).

If you use this dataset, please cite the following paper:    

[Bar-Haim et al., 2017a] 
Roy Bar-Haim, Indrajit Bhattacharya, Francesco Dinuzzo, Amrita Saha, and Noam Slonim. 2017. 
Stance Classification of Context-Dependent Claims.
In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: 
Volume 1, Long Papers. Association for Computational Linguistics, Valencia, Spain, pages 251–261.

Improved stance classification results on this dataset were published in:

[Bar-Haim et al., 2017b] 
Roy Bar-Haim, Lilach Edelstein, Charles Jochim and Noam Slonim. 
Improving Claim Stance Classification with Lexical Knowledge Expansion and Context Utilization. 2017. 
In Proceedings of the 4th Workshop on Argument Mining. 
Association for Computational Linguistics, Copenhagen, Denmark, pages 32-38.

The datasets are released under the following licensing and copyright terms:
• (c) Copyright Wikipedia (https://en.wikipedia.org/wiki/Wikipedia:Copyrights#Reusers.27_rights_and_obligations)
• (c) Copyright IBM 2014. Released under CC-BY-SA (http://creativecommons.org/licenses/by-sa/3.0/)


CONTENTS

The JSON file claim_stance_dataset_v1.json has the following structure (attribute order may change in the actual file):

[
  {
    "topicId": 1,  // internal topic ID
    "split": "test", // train or test
    "topicText": "This house believes that the sale of violent video games to minors should be banned",
    "topicTarget": "the sale of violent video games to minors",  // sentiment target of topic
    "topicSentiment": -1  // topic sentiment towards its target (1:positive/-1:negative)
    "claims": [
      {
        "claimId": 2973, // claim internal ID
        "stance": "PRO", // PRO or CON
        // the corrected version of the claim
        "claimCorrectedText": "Exposure to violent video games causes at least a temporary increase in aggression and this exposure correlates with aggression in the real world",
        // the original version of the claim
        "claimOriginalText": "exposure to violent video games causes at least a temporary increase in aggression and that this exposure correlates with aggression in the real world",
 
        // the article containing the claim    
        "article": {
          "rawFile": "articles/t1/raw_1.txt" //raw version of article
          "rawSpan": { // claim's span in raw file
            "start": 490,
            "end": 640
          },
          "cleanFile": "articles/t1/clean_1.txt", //clean version of article
          "cleanSpan": { // claim's span in clean file
            "start": 418,
            "end": 568
          },
        }
        
        "Compatible": "yes", // is the claim compatible with the semantic model of Bar-Haim et al. [2017a]? (yes/no)
        
        // the following fine-grained annotations are specified only for "compatible" claims
        "claimTarget": {  // claim sentiment target (in the corrected version of the claim)
          "text": "Exposure to violent video games",
          "span": {
            "start": 0,
            "end": 31
          }
        },
        "claimSentiment": -1, //claim's sentiment towards its target (1:positive/-1:negative)
        "targetsRelation": 1, // relation between claim target and topic target ((1:consistent/-1:contrastive))
      },
      ... // additional claims for topic
    ],
  },
  {
... // additional topics
]

The CSV file claim_stance_dataset_v1.csv contains the same information in CSV format.

The CSV file article_info.csv includes the following columns for each article:
1. topic_id 
2. raw_file - raw file path
3. clean_file - clean file path
4. title - article title
5. url - link to source Wikipedia article

NOTES:
(1) Claim annotations and the experiments reported in [Bar-Haim et al., 2017a] and [Bar-Haim et al., 2017b] 
    are based on the corrected version of the claim. See [Aharoni et al., 2014] for description of generating 
    corrected version for claims. The original version is the claim as it is found in the clean version of 
    the article, with no further editing.

   [Aharoni et al., 2014]
    A Benchmark Dataset for Automatic Detection of Claims and Evidence in the Context of Controversial Topics 
    Ehud Aharoni, Anatoly Polnarov, Tamar Lavee, Daniel Hershcovich, Ran Levy, Ruty Rinott, Dan Gutfreund, 
    and Noam Slonim
    Proceedings of the First Workshop on Argumentation Mining, ACL, pp. 64-68, 
    Association for Computational Linguistics, 2014

(2) The topics and claims partially overlap with the CE-EMNLP-2015 dataset:
    Common topics IDs: 1, 21, 61, 81, 101, 121, 181, 221, 323, 381, 441, 442, 443, 481, 482, 483, 601, 602, 
    621, 641, 642, 644, 645, 648, 662, 663, 665, 681, 683, 701, 721, 742, 743, 744, 761, 801, 803, 841, 861, 
    881, 923, 926, 941, 942, 944, 946
    Only this dataset: 603, 661, 922, 985, 987, 990, 994, 1005, 1065
    Only the CE-EMNLP-2015 dataset: 643, 646, 647, 664, 821, 902, 921, 925, 943, 945, 947, 961
