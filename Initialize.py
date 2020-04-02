import GetProperties as gp
import csv


def create_patient_population(num_patients=10000, csv_name="population", stages=["1", "2", "3A", "3B"]):

    dir_list = list()

    for stage in stages:

        if (stage == "1" or stage == "2"):
            pop = gp.sample_dist(int(stage), num_patients)
            # pop = gp.get_tumor_diameter(pop)

            with open(csv_name + stage + ".csv", mode='w') as f:
                writer = csv.writer(f, delimiter=',')

                for num in range(0, num_patients):
                    writer.writerow([num, pop[num]])

        else:
            pop = gp.sample_dist(3, num_patients)
            # pop = gp.get_tumor_diameter(pop)

            with open(csv_name + stage + ".csv", mode='w') as f:
                writer = csv.writer(f, delimiter=',')

                for num in range(0, num_patients):
                    writer.writerow([num, pop[num]])

        dir_list.append("./" + csv_name + stage + ".csv")
    
    return 