# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#


"""
Import Brain Tumor dataset

.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""
import sys
import uuid
import csv
import numpy as np
import json

from tvb.adapters.datatypes.db.graph import CorrelationCoefficientsIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.graph_h5 import CorrelationCoefficientsH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.adapters.uploaders.csv_connectivity_importer import DELIMITER_OPTIONS
from tvb.adapters.uploaders.region_mapping_importer import RegionMappingImporter, RegionMappingImporterModel
from tvb.adapters.uploaders.zip_surface_importer import ZIPSurfaceImporter, ZIPSurfaceImporterModel
from tvb.basic.logger.builder import get_logger
from tvb.basic.readers import try_get_absolute_path
from tvb.config.algorithm_categories import DEFAULTDATASTATE_INTERMEDIATE
from tvb.core.adapters.abcuploader import ABCUploader
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.datatypes.graph import CorrelationCoefficients
from tvb.datatypes.time_series import TimeSeries, TimeSeriesRegion
from tvb.interfaces.command.lab import *
from tvb.tests.framework.core.factory import TestFactory

CONN_ZIP_FILE = "SC.zip"
FC_MAT_FILE = "FC.mat"
FC_DATASET_NAME = "FC_cc_DK68"
TIME_SERIES_CSV_FILE = "HRF.csv"

LOG = get_logger(__name__)


def prepare_tumor_connectivity(conn_folder, patient, user_tag):
    connectivity_zip = os.path.join(conn_folder, CONN_ZIP_FILE)
    if not os.path.exists(connectivity_zip):
        LOG.error("File {} does not exist.".format(connectivity_zip))
        return
    import_conn_adapter = ABCAdapter.build_adapter_from_class(ZIPConnectivityImporter)
    import_conn_model = ZIPConnectivityImporterModel()
    import_conn_model.uploaded = connectivity_zip
    import_conn_model.data_subject = patient
    import_conn_model.generic_attributes.user_tag_1 = user_tag

    return import_conn_adapter, import_conn_model


def import_tumor_datatype(project_id, adapter, model):
    import_op = fire_operation(project_id, adapter, model)
    import_op = wait_to_finish(import_op)

    return dao.get_results_for_operation(import_op.id)[0].gid


def import_time_series_csv_datatype(hrf_folder, project_id, connectivity_gid, patient, user_tag):
    path = os.path.join(hrf_folder, TIME_SERIES_CSV_FILE)
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=DELIMITER_OPTIONS['comma'])
        ts = list(csv_reader)

    ts_data = np.array(ts, dtype=np.float64).reshape((len(ts), 1, len(ts[0]), 1))
    ts_time = np.random.rand(ts_data.shape[0],)

    project = dao.get_project_by_id(project_id)
    user = dao.get_user_by_id(project.fk_admin)
    op = TestFactory.create_operation(user, project)

    ts_gid = uuid.uuid4()
    h5_path = "TimeSeries_{}.h5".format(ts_gid.hex)
    operation_folder = FilesHelper().get_operation_folder(project.name, op.id)
    h5_path = os.path.join(operation_folder, h5_path)

    conn = h5.load_from_gid(connectivity_gid)
    ts = TimeSeriesRegion()
    ts.data = ts_data
    ts.time = ts_time
    ts.gid = ts_gid
    ts.connectivity = conn
    generic_attributes = GenericAttributes()
    generic_attributes.user_tag_1 = user_tag
    generic_attributes.state = DEFAULTDATASTATE_INTERMEDIATE

    with TimeSeriesRegionH5(h5_path) as ts_h5:
        ts_h5.store(ts)
        ts_h5.nr_dimensions.store(4)
        ts_h5.subject.store(patient)
        ts_h5.store_generic_attributes(generic_attributes)

    ts_index = TimeSeriesIndex()
    ts_index.gid = ts_gid.hex
    ts_index.fk_from_operation = op.id
    ts_index.time_series_type = "TimeSeriesRegion"
    ts_index.data_length_1d = ts_data.shape[0]
    ts_index.data_length_2d = ts_data.shape[1]
    ts_index.data_length_3d = ts_data.shape[2]
    ts_index.data_length_4d = ts_data.shape[3]
    ts_index.data_ndim = len(ts_data.shape)
    ts_index.sample_period_unit = 'ms'
    ts_index.sample_period = TimeSeries.sample_period.default
    ts_index.sample_rate = 1024.0
    ts_index.subject = patient
    ts_index.state = DEFAULTDATASTATE_INTERMEDIATE
    ts_index.labels_ordering = json.dumps(list(TimeSeries.labels_ordering.default))
    ts_index.labels_dimensions = json.dumps(TimeSeries.labels_dimensions.default)
    dao.store_entity(ts_index)

    return ts_gid


def import_pearson_coefficients_datatype(fc_folder, project_id, ts_gid, patient, user_tag):
    path = os.path.join(fc_folder, FC_MAT_FILE)
    result = ABCUploader.read_matlab_data(path, FC_DATASET_NAME)
    result = result.reshape((result.shape[0], result.shape[1], 1, 1))

    project = dao.get_project_by_id(project_id)
    user = dao.get_user_by_id(project.fk_admin)
    op = TestFactory.create_operation(user, project)

    pearson_gid = uuid.uuid4()
    h5_path = "CorrelationCoefficients_{}.h5".format(pearson_gid.hex)
    operation_folder = FilesHelper().get_operation_folder(project.name, op.id)
    h5_path = os.path.join(operation_folder,  h5_path)

    generic_attributes = GenericAttributes()
    generic_attributes.user_tag_1 = user_tag
    generic_attributes.state = DEFAULTDATASTATE_INTERMEDIATE

    with CorrelationCoefficientsH5(h5_path) as pearson_correlation_h5:
        pearson_correlation_h5.array_data.store(result)
        pearson_correlation_h5.gid.store(pearson_gid)
        pearson_correlation_h5.parent_burst.store(pearson_gid)
        pearson_correlation_h5.source.store(ts_gid)
        pearson_correlation_h5.labels_ordering.store(CorrelationCoefficients.labels_ordering.default)
        pearson_correlation_h5.subject.store(patient)
        pearson_correlation_h5.store_generic_attributes(generic_attributes)

    pearson_correlation_index = CorrelationCoefficientsIndex()
    pearson_correlation_index.fk_source_gid = ts_gid.hex
    pearson_correlation_index.gid = pearson_gid.hex
    pearson_correlation_index.fk_from_operation = op.id
    pearson_correlation_index.subject = patient
    pearson_correlation_index.state = DEFAULTDATASTATE_INTERMEDIATE
    pearson_correlation_index.ndim = 4
    dao.store_entity(pearson_correlation_index)

    return ts_gid


def import_tumor_datatypes(project_id, folder_path):
    conn_gids = []
    for patient in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient)
        if os.path.isdir(patient_path):
            user_tags = os.listdir(patient_path)
            for user_tag in user_tags:
                datatype_folder = os.path.join(patient_path, user_tag)

                conn_adapter, conn_model = prepare_tumor_connectivity(datatype_folder, patient, user_tag)
                conn_gid = import_tumor_datatype(project_id, conn_adapter, conn_model)

                ts_gid = import_time_series_csv_datatype(datatype_folder, project_id, conn_gid, patient, user_tag)
                import_pearson_coefficients_datatype(datatype_folder, project_id, ts_gid, patient, user_tag)

                conn_gids.append(conn_gid)
    return conn_gids


def import_surface_rm(project_id, conn_gid):
    # Import surface and region mapping from tvb_data berlin subjects (68 regions)
    rm_file = try_get_absolute_path("tvb_data", "berlinSubjects/DH_20120806/DH_20120806_RegionMapping.txt")
    surface_zip_file = try_get_absolute_path("tvb_data", "berlinSubjects/DH_20120806/DH_20120806_Surface_Cortex.zip")

    surface_importer = ABCAdapter.build_adapter_from_class(ZIPSurfaceImporter)
    surface_imp_model = ZIPSurfaceImporterModel()
    surface_imp_model.uploaded = surface_zip_file
    surface_imp_operation = fire_operation(project_id, surface_importer, surface_imp_model)
    surface_imp_operation = wait_to_finish(surface_imp_operation)

    surface_gid = dao.get_results_for_operation(surface_imp_operation.id)[0].gid
    rm_importer = ABCAdapter.build_adapter_from_class(RegionMappingImporter)
    rm_imp_model = RegionMappingImporterModel()
    rm_imp_model.mapping_file = rm_file
    rm_imp_model.surface = surface_gid
    rm_imp_model.connectivity = conn_gid
    rm_import_operation = fire_operation(project_id, rm_importer, rm_imp_model)
    wait_to_finish(rm_import_operation)


if __name__ == '__main__':
    # Path to TVB folder
    input_folder = sys.argv[1]
    # Project where DTs will be imported
    project_id = sys.argv[2]

    conn_gids = import_tumor_datatypes(project_id, input_folder)
    import_surface_rm(project_id, conn_gids[0])
