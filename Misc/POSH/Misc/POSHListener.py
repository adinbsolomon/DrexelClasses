# Generated from POSH.g4 by ANTLR 4.9.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .POSHParser import POSHParser
else:
    from POSHParser import POSHParser

    
stack = []

# This class defines a complete listener for a parse tree produced by POSHParser.
class POSHListener(ParseTreeListener):

    # Enter a parse tree produced by POSHParser#requirement.
    def enterRequirement(self, ctx:POSHParser.RequirementContext):
        print('entering requirement')

    # Exit a parse tree produced by POSHParser#requirement.
    def exitRequirement(self, ctx:POSHParser.RequirementContext):
        if ctx.CONCURRENCY():
            stack.append("CONCURRENCY")
        print(stack)
        print('exiting requirement')


    # Enter a parse tree produced by POSHParser#course.
    def enterCourse(self, ctx:POSHParser.CourseContext):
        print('entering course')

    # Exit a parse tree produced by POSHParser#course.
    def exitCourse(self, ctx:POSHParser.CourseContext):
        # previous two bois are subject and number
        stack.append(f"{stack.pop(-2)} {stack.pop(-1)}")
        print(stack)
        print('exiting course')


    # Enter a parse tree produced by POSHParser#subject.
    def enterSubject(self, ctx:POSHParser.SubjectContext):
        print('entering subject')

    # Exit a parse tree produced by POSHParser#subject.
    def exitSubject(self, ctx:POSHParser.SubjectContext):
        i = 0
        s = ''
        while c := ctx.LETTER(i):
            s += c.__str__()
            i+=1
        stack.append(s)
        print(stack)
        print('exiting subject')


    # Enter a parse tree produced by POSHParser#number.
    def enterNumber(self, ctx:POSHParser.NumberContext):
        print('entering number')

    # Exit a parse tree produced by POSHParser#number.
    def exitNumber(self, ctx:POSHParser.NumberContext):
        s = ''
        if c := ctx.SPECIAL_DIGIT():
            s += c.__str__()
        i = 0
        while c := ctx.DIGIT(i):
            s += c.__str__()
            i+=1
        stack.append(s)
        print(stack)
        print('exiting number')


    # Enter a parse tree produced by POSHParser#min_grade.
    def enterMin_grade(self, ctx:POSHParser.Min_gradeContext):
        print('entering min_grade')

    # Exit a parse tree produced by POSHParser#min_grade.
    def exitMin_grade(self, ctx:POSHParser.Min_gradeContext):
        i = 0
        s = ''
        while c := ctx.LETTER(i):
            s += c.__str__()
            i += 1
        stack.append(s)
        print(stack)
        print('exiting min_grade')



del POSHParser