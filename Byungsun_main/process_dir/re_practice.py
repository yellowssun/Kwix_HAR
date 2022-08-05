import re

regex = r"\d\d\d-\d-\d-02-\S\d_\S"


test_str = ("018-1-1-01-Z4_C\n"
	"019-1-1-03-Z4_C\n"
	"020-1-1-02-Z4_C\n"
	"024-1-1-01-Z4_C\n"
    "020-1-1-02-Z4_C\n"
    "020-1-1-02-Z4_C\n"
    "020-1-1-02-Z4_C\n"
    "020-1-1-02-Z4_C")

print(type(test_str))

a = [1, 2, 3, 4]
matches = re.findall(regex, test_str, re.MULTILINE)
matches = matches
print(type(matches), matches)


# for matchNum, match in enumerate(matches, start=1):
    
#     print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
    
#     for groupNum in range(0, len(match.groups())):
#         groupNum = groupNum + 1
        
#         print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))