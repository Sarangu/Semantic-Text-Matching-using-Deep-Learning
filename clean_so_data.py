import pandas as pd
import re
import pickle as pkl
from convert_so_csv import LINK_CSV_PATH, POST_CSV_PATH
import csv

posts_out_path = "../formatted_data/stackoverflow/posts_with_links.csv"
linked_out_path = "../formatted_data/stackoverflow/related_posts.csv"

post_links = pd.read_csv(LINK_CSV_PATH, sep="|")
post_links = post_links[post_links["LinkTypeId"] == 3]


def clean_body(t):
    return re.sub(" +", " ", re.sub("\<.+?\>", "", t).replace("\n", " "))

if __name__ == "__main__":
  with open(POST_CSV_PATH, "r") as inf, open(posts_out_path, "w") as posts_outf, open(linked_out_path, "w") as linked_outf:
      reader = csv.DictReader(inf, delimiter="|")
      post_writer = None
      linked_writer = None
      for i, row in enumerate(reader):
          if i == 0:
              fieldnames = row.keys()
              post_writer = csv.DictWriter(posts_outf, delimiter="|", fieldnames=fieldnames)
              linked_writer = csv.DictWriter(linked_outf, delimiter="|", fieldnames=fieldnames)
              post_writer.writeheader()
              linked_writer.writeheader()
          if i % 100000 == 0:
              print(i, row["Id"], row["CreationDate"])
          if int(row["Id"]) in orig_posts:
              post_writer.writerow(row)
          if int(row["Id"]) in related_posts:
              linked_writer.writerow(row)


  short_posts_df = pd.read_csv(posts_out_path, sep="|")
  short_linked_df = pd.read_csv(linked_out_path, sep="|")
  short_posts_df["Body"] = (
      short_posts_df["Body"]
      .str.replace(re.compile("\<.+?\>"), "")
      .str.replace("\n", " ")
      .str.replace(re.compile(" +"), " ")
  )
  true_links = post_links[["PostId", "RelatedPostId"]]

  linked_posts = pd.merge(true_links, short_posts_df, left_on=["PostId"], right_on=["Id"], how="inner")
  linked_posts = linked_posts[["PostId", "RelatedPostId", "Title", "Body", "CreationDate"]].rename(
      columns={"Title": "PostTitle", "Body": "PostBody", "CreationDate": "PostDate"}
  )

  linked_posts = pd.merge(linked_posts, short_linked_df, left_on=["RelatedPostId"], right_on=["Id"], how="left")
  linked_posts = linked_posts[["PostId", "RelatedPostId", "PostTitle", "PostBody", "PostDate", "Title", "Body", "CreationDate"]].rename(
      columns={"Title": "RelatedTitle", "Body": "RelatedBody", "CreationDate": "RelatedDate"}
  )

  link_data_basic = [(row["PostTitle"], row["RelatedTitle"]) for _, row in linked_posts.iterrows()]
  pkl.dump(link_data_basic, open("../formatted_data/stackoverflow/q_link_pairs_titles_only.pkl", "wb"))

  answer_title_body_link = [
      (row["RelatedTitle"], str(row["RelatedTitle"]) + " " + clean_body(str(row["RelatedBody"]))) for _, row in linked_posts.iterrows()
  ]
  pkl.dump(answer_title_body_link, open("../formatted_data/stackoverflow/answer_title_body_lookup.pkl", "wb"))

  unique_answer_titles = list(set([a for (_, a) in link_data_basic]))
  pkl.dump(link_data_basic, open("../formatted_data/stackoverflow/unique_answer_titles.pkl", "wb"))
