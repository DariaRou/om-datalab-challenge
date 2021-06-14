from om-datalab-challenge.data import om_get_data

def test_om_get_data():

    data = om_get_data()
    assert type(data) != None
