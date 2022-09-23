#!/usr/bin/env python3

def add_trajectory_to_index(conn):
    flights_2202R = {
        "air_steps": ['01', '02', '04', '05', '06', '07', '09', '10', '11', '12', '13'],
        "ground_flight": ['08', '14', '15'],
        "ground_stationary": ['03'],
    }
    flights_2202U = {
        "air_steps": ['03', '04', '05', '06', '07', '08', '10', '11', '12', '13', '14', '15'],
        "ground_flight": ['16', '17', '18', '19'],
        "ground_stationary_indoors": ['01', '02'],
        "ground_stationary_outdoors": ['09'],
    }
    c = conn.cursor()
    for flight_name, flights in [('2202R', flights_2202R), ('2202U', flights_2202U)]:
        for trajectory, flights in flights.items():
            for flight in flights:
                sql = f'''UPDATE "index" SET "trajectory" = '{trajectory}'
                    WHERE trajectory IS NULL AND "day" = '{flight_name}' AND "flight" = '{flight}'
                '''
                print(sql)
                c.execute(sql)
                conn.commit()


# def query_id_set(conn, ids, query):
#     sql = f"{query} "
