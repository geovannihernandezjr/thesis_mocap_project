"""
author: Geovanni Hernandez
Separate method to remove unwanted data from the motion capture TSV file.
"""

def remove_markers_not_needed_in_each_df_and_marker_name_lst(mocap_multi_df, moCapDataAttributes):
    '''

    :param mocap_multi_df: the multidataframe that contains 2 level heading data
    :param moCapDataAttributes: attributes of the mocap file
    :return: the dataframe with removed data
    '''
    if 'DestLeftLower' in mocap_multi_df.columns:  # delete columns not needed and markernames from master list of names
        del mocap_multi_df['DestLeftLower']
        del mocap_multi_df['DestLeftUpper']
        del mocap_multi_df['DestRightUpper']
        del mocap_multi_df['DestRightLower']

        moCapDataAttributes.markerNames.remove('DestLeftLower')
        moCapDataAttributes.markerNames.remove('DestLeftUpper')
        moCapDataAttributes.markerNames.remove('DestRightUpper')
        moCapDataAttributes.markerNames.remove('DestRightLower')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'DestLeftLower' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'DestLeftUpper' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'DestRightLower' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'DestRightUpper' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'LDestLower' in mocap_multi_df.columns:  # delete columns not needed and markernames from master list of names
        del mocap_multi_df['LDestLower']
        del mocap_multi_df['LDestUpper']
        del mocap_multi_df['RDestLower']
        del mocap_multi_df['RDestUpper']

        moCapDataAttributes.markerNames.remove('LDestLower')
        moCapDataAttributes.markerNames.remove('LDestUpper')
        moCapDataAttributes.markerNames.remove('RDestLower')
        moCapDataAttributes.markerNames.remove('RDestUpper')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LDestLower' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'LDestUpper' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RDestLower' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RDestUpper' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'LForefootIn' in mocap_multi_df.columns:
        del mocap_multi_df['LForefootIn']
        del mocap_multi_df['RForefootIn']


        moCapDataAttributes.markerNames.remove('LForefootIn')
        moCapDataAttributes.markerNames.remove('RForefootIn')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LForefootIn' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RForefootIn' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'LForefootOut' in mocap_multi_df.columns:
        del mocap_multi_df['LForefootOut']
        del mocap_multi_df['RForefootOut']

        moCapDataAttributes.markerNames.remove('LForefootOut')
        moCapDataAttributes.markerNames.remove('RForefootOut')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LForefootOut' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RForefootOut' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'HeadTop' in mocap_multi_df.columns:
        del mocap_multi_df['HeadTop']
        del mocap_multi_df['HeadL']
        del mocap_multi_df['HeadR']

        moCapDataAttributes.markerNames.remove('HeadTop')
        moCapDataAttributes.markerNames.remove('HeadL')
        moCapDataAttributes.markerNames.remove('HeadR')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'HeadTop' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'HeadR' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'HeadL' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
    if 'SpineTop' in mocap_multi_df.columns:
        del mocap_multi_df['SpineTop']
        moCapDataAttributes.markerNames.remove('SpineTop')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'SpineTop' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
    if 'HeadFront' in mocap_multi_df.columns:
        del mocap_multi_df['HeadFront']

        moCapDataAttributes.markerNames.remove('HeadFront')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'HeadFront' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)


    ### add new code starting here
    if 'BackMid' in mocap_multi_df.columns:
        del mocap_multi_df['BackTop']
        del mocap_multi_df['BackMid']
        del mocap_multi_df['BackLow']

        moCapDataAttributes.markerNames.remove('BackTop')
        moCapDataAttributes.markerNames.remove('BackMid')
        moCapDataAttributes.markerNames.remove('BackLow')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'BackTop' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'BackMid' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'BackLow' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'LHandOut' in mocap_multi_df.columns:
        del mocap_multi_df['LHandOut']
        del mocap_multi_df['RHandOut']
        moCapDataAttributes.markerNames.remove('LHandOut')
        moCapDataAttributes.markerNames.remove('RHandOut')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LHandOut' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RHandOut' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'LHeelBack' in mocap_multi_df.columns:
        del mocap_multi_df['LHeelBack']
        del mocap_multi_df['RHeelBack']
        moCapDataAttributes.markerNames.remove('LHeelBack')
        moCapDataAttributes.markerNames.remove('RHeelBack')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LHeelBack' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RHeelBack' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)


    if 'LToeTip' in mocap_multi_df.columns:
        del mocap_multi_df['LToeTip']
        del mocap_multi_df['RToeTip']

        moCapDataAttributes.markerNames.remove('LToeTip')
        moCapDataAttributes.markerNames.remove('RToeTip')

        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LToeTip' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RToeTip' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'LAnkleOut' in mocap_multi_df.columns:
        del mocap_multi_df['LAnkleOut']
        del mocap_multi_df['RAnkleOut']

        moCapDataAttributes.markerNames.remove('LAnkleOut')
        moCapDataAttributes.markerNames.remove('RAnkleOut')

        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LAnkleOut' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RAnkleOut' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'WaistLFront' in mocap_multi_df.columns:
        del mocap_multi_df['WaistLFront']
        del mocap_multi_df['WaistRFront']

        moCapDataAttributes.markerNames.remove('WaistLFront')
        moCapDataAttributes.markerNames.remove('WaistRFront')

        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'WaistLFront' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'WaistRFront' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'Hips_WaistLFront' in mocap_multi_df.columns:
        del mocap_multi_df['Hips_WaistLFront']
        del mocap_multi_df['Hips_WaistRFront']
        moCapDataAttributes.markerNames.remove('Hips_WaistLFront')
        moCapDataAttributes.markerNames.remove('Hips_WaistRFront')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'Hips_WaistLFront' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'Hips_WaistRFront' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)


    if 'LArm' in mocap_multi_df.columns:
        del mocap_multi_df['LArm']
        moCapDataAttributes.markerNames.remove('LArm')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LArm' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)


    if 'RArm' in mocap_multi_df.columns:
        del mocap_multi_df['RArm']
        moCapDataAttributes.markerNames.remove('RArm')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'RArm' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)


    if 'Chest' in mocap_multi_df.columns:
        del mocap_multi_df['Chest']
        moCapDataAttributes.markerNames.remove('Chest')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'Chest' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'LWristIn' in mocap_multi_df.columns:
        del mocap_multi_df['LWristIn']
        del mocap_multi_df['RWristIn']
        moCapDataAttributes.markerNames.remove('LWristIn')
        moCapDataAttributes.markerNames.remove('RWristIn')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LWristIn' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RWristIn' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'LWristOut' in mocap_multi_df.columns:
        del mocap_multi_df['LWristOut']
        del mocap_multi_df['RWristOut']
        moCapDataAttributes.markerNames.remove('LWristOut')
        moCapDataAttributes.markerNames.remove('RWristOut')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'LWristOut' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'RWristOut' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    if 'WaistLBack' in mocap_multi_df.columns:
        del mocap_multi_df['WaistLBack']
        del mocap_multi_df['WaistRBack']
        moCapDataAttributes.markerNames.remove('WaistLBack')
        moCapDataAttributes.markerNames.remove('WaistRBack')
        for element in moCapDataAttributes.markerNamesXYZ.copy():
            if 'WaistLBack' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)
            elif 'WaistRBack' in element:
                moCapDataAttributes.markerNamesXYZ.remove(element)

    return mocap_multi_df, moCapDataAttributes
