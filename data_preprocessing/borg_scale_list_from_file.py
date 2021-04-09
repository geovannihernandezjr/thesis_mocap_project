"""
Author: Geovanni Hernandez
Date: 2/1/2020
Purpose: To create a solution to get borg scale data to use for merging to mocap data. This uses xlrd to open xlsx file and each
rpe value obtained every minutes.
Note: XLRD removed support for excel files. Using previous versin pip install xlrd==1.2.0 works but need to modify code to probably use openpyxl or pandas
"""
import xlrd  ## Using xlrd package to open xlsx file so i do not have to convert excel file to txt manually


from numpy import array, int16
def get_borgscale_list(file):
    """           BORG FILE OPEN            """
    # Give the location of the file
    loc = file  # ("Borg_SP_25_16_14_9-20-2019.xlsx")

    # To open Workbook
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)  # first sheet page in the excel file

    if 'minute' in sheet.cell_value(0, 0).lower() and 'borgs scale' in sheet.cell_value(0, 1).replace("'", "").lower():
        borg = list(map(int, sheet.col_values(1)[1:]))
    else:
        borg = list(map(int, sheet.col_values(1)[:]))
    # print(borg)
    """           BORG FILE CLOSE           """
    return array(borg, int16)
