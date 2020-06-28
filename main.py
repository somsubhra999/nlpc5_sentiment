from pathlib import Path
import csv
import ngram_toolkit as ngtk

inpf_path = Path(Path.cwd(), 'data/ticket_Data.csv')
outf_path = Path(Path.cwd(), 'data/ticket_Data_freq.csv')
has_header = True          # Change this to False if the csv file doesn't have a header

description_list = []

with open(inpf_path, 'r') as inp_csvfile:
    ticket_rdr = csv.reader(inp_csvfile, delimiter=',', quotechar='"')

    if has_header:
        header = next(ticket_rdr)

    for row in ticket_rdr:
        ticket_id, description = row
        description = description.strip()
        if description:
            description_list.append(description)

full_ngram = ngtk.Ngram('\n'.join(description_list))
full_fdist_1gram = full_ngram.get_fdist(1)

# Keeping only the words which have frequency more than 1
non_unique = list(filter(lambda x: x[1] > 1, full_fdist_1gram.items()))
words = [x[0] for x in [*non_unique]]

with open(outf_path, 'w', newline='\n', encoding='utf-8') as out_csvfile:
    csvwriter = csv.writer(out_csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Writing header row
    csvwriter.writerow(['Description', 'Label', *words])

    for description in description_list:
        curr_ngram = ngtk.Ngram(description)
        curr_fdist_1gram = curr_ngram.get_fdist(1)

        curr_vocab = []
        for word in words:
            if word in curr_fdist_1gram:
                curr_vocab.append(curr_fdist_1gram[word])
            else:
                curr_vocab.append(0)

        polarity = ngtk.Ngram.get_polarity(description)
        label = 'info' if polarity['compound'] >= 0 else 'issue'

        csvwriter.writerow([description, label, *curr_vocab])
