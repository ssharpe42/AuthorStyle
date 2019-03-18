from Document import Document

class Corpus():

    def __init__(self):

        self.documents = []

    def init_docs(self):

        """Initialize and process documents"""

        for i in range(len(self.documents)):
            self.documents[i].process_doc()

    def add_document(self, document):

        assert isinstance(document, Document), "Add only documents"

        self.documents.append(document)

    def create_vocab(self):
        pass

    #....etc

    def features_to_dataframe(self):
        pass