
import Database

#Database.update_navigation()

def update_subject(subject):
    Database.update_subject(subject.upper())
    print(f"\n\n\nDone with {subject.upper()}\n\n\n")

update_subject('MGMT')
