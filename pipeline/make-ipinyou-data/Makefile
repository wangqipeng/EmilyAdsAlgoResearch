
# makefile to create ipinyou dataset evaluation
BASE=.
ORIGINALFOLDER=./original-data/ipinyou.contest.dataset
TRAIN=$(ORIGINALFOLDER)/conv_train
TEST=$(ORIGINALFOLDER)/conv_test

all: init conv train.log test.log advertisers yzx

init: $(ORIGINALFOLDER)
	mkdir -p $(TRAIN)
	cp $(ORIGINALFOLDER)/training2nd/clk.*.txt $(TRAIN)
	cp $(ORIGINALFOLDER)/training2nd/conv.*.txt $(TRAIN)
	cp $(ORIGINALFOLDER)/training3rd/clk.*.txt $(TRAIN)
	cp $(ORIGINALFOLDER)/training3rd/conv.*.txt $(TRAIN)
	#bzip2 -d $(TRAIN)/*
	mkdir -p $(TEST)
	cp $(ORIGINALFOLDER)/testing2nd/* $(TEST)
	cp $(ORIGINALFOLDER)/testing3rd/* $(TEST)
	#bzip2 -d $(TEST)/*
	mkdir $(BASE)/conv_all	

conv: $(TRAIN)
	cat $(TRAIN)/conv*.txt > $(BASE)/conv_all/conv.all.txt
	
#click log join conv log
train.log: $(BASE)/schema.txt $(BASE)/conv_all/conv.all.txt 
	cat $(TRAIN)/clk*.txt | $(BASE)/python/mkdata.py $+ > $(BASE)/conv_all/train.log.txt
	$(BASE)/python/formalizeua.py $(BASE)/conv_all/train.log.txt

test.log: $(BASE)/schema.txt
	cat $(TEST)/*.txt | $(BASE)/python/mktest.py $+ > $(BASE)/conv_all/test.log.txt
	$(BASE)/python/formalizeua.py $(BASE)/conv_all/test.log.txt

advertisers: $(BASE)/conv_all/train.log.txt $(BASE)/conv_all/test.log.txt
	$(BASE)/python/splitadvertisers.py $(BASE) 25 $(BASE)/conv_all/train.log.txt $(BASE)/conv_all/test.log.txt

yzx: advertisers
	bash $(BASE)/mkyzxdata.sh


