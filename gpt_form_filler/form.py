import datetime
from typing import Any, List, Optional, Tuple

import phonenumbers
import pytz
from dateutil import parser


def get_local_timezone() -> pytz.tzinfo:
    return datetime.datetime.now().astimezone().tzinfo


class Option:
    def __init__(self, label, value):
        self.label = label
        self.value = value


# Treat this as a form field
# TODO(P0, devx): Make field_type an Enum
# TODO: Custom display transformations;
class FieldDefinition:
    def __init__(
        self,
        name: str,
        # text, date, html, number, phonenumber, bool, select
        field_type: str,
        label: str,
        description: Optional[str] = None,
        options: Optional[List[Option]] = None,
        ignore_in_prompt: bool = False,
        ignore_in_display: bool = False,
        ignore_in_email: bool = False,
        custom_field: Optional[bool] = None,
        default_value: Any = None,
    ):
        self.name = name
        self.field_type = field_type
        self.label = label
        self.description = description
        self.options = options
        self.ignore_in_prompt = ignore_in_prompt
        self.ignore_in_display = ignore_in_display
        self.ignore_in_email = ignore_in_email
        self.custom_field = bool(custom_field)
        self.default_value = default_value

    @classmethod
    def _none_or_quoted_str(cls, value: Optional[str]) -> str:
        return f'"{value}"' if isinstance(value, str) else "None"

    @classmethod
    def _gen_options(cls, options: Optional[list[Option]]) -> str:
        if isinstance(options, list):
            options_definitions = [
                f'Option(label="{opt.label}", value="{opt.value}")' for opt in options
            ]
            return "[" + ",\n".join(options_definitions) + "]"
        return "None"

    # Code-gen new forms - handy to do from e.g. APIs.
    def to_python_definition(self) -> str:
        return """
        "{name}": FieldDefinition(
            name="{name}",
            field_type="{field_type}",
            label="{label}",
            description={description},
            options={options},
            custom_field={custom_field}
        )""".format(
            name=self.name,
            field_type=self.field_type,
            label=self.label,
            description=FieldDefinition._none_or_quoted_str(self.description),
            options=FieldDefinition._gen_options(self.options),
            custom_field=str(self.custom_field),
        )

    def _has_options(self):
        return self.field_type in ["radio", "select"]

    # TODO(ux, p1): Feels like the display_value should be outside of form.py, cause it depends on the output dest
    #   being like email, spreadsheet, app or webapp. Database / Python kinda counts too.
    def display_value(self, value):
        if self.ignore_in_display:
            print(
                f"WARNING: display_value called for ignore_in_display field {self.name} value {str(value)[:100]}"
            )
            return "Hidden"

        if value is None:
            # TODO(P0, hack): Add a FieldDefinition.required or some display_value transformations
            # if self.name in ["name", "phone"]:
            #     return "None - Please fill in"
            return "None"

        if self._has_options():
            option_labels = {option.value: option.label for option in self.options}
            return option_labels.get(value, str(value))

        if self.field_type == "date":
            datetime_value = self._validate_date(value)

            # Convert the datetime to PST
            pst = pytz.timezone("America/Los_Angeles")
            datetime_value = datetime_value.astimezone(pst)

            # We output more machine-like so it can be sorted in Excel
            return datetime_value.strftime("%Y-%m-%d %H:%M %Z")

        # TODO(P0, hack): Add special handling on the FieldDefinition; especially HubSpot fields like ObjId can use it
        # if self.name in ["firstname", "lastname"]:
        #     return " ".join(word.capitalize() for word in str(value).split(" "))

        return value

    def validate_and_fix(self, value: Optional[Any]) -> Any:
        if value is None:
            return None

        if str(value).lower() in ["none", "null", "unknown"]:
            return None

        # Sometimes, GPT results in entire definition of the field, in that case extra the value
        # For example:
        # Invalid format for task.hs_task_subject expected unexpected text type (type=text) given
        # { 'label': 'Task Title',
        #   'description': 'The title of the task',
        #   'type': 'text',
        #   'value': 'Schedule meeting with Andrey Yursa'
        # } (type=<class 'dict'>)
        if isinstance(value, dict):
            # NOTE: We also include "type" cause ("label", "value") happens often for Select/Radio/Option fields.
            if "label" in value and "type" in value and "value" in value:
                return self.validate_and_fix(value["value"])

        if self.field_type == "text":
            return self._validate_text(value)
        if self.field_type == "date":
            return self._validate_date(value)
        if self.field_type == "html":
            return self._validate_html(value)
        if self.field_type == "number":
            return self._validate_number(value)
        if self.field_type == "phonenumber":
            return self._validate_phonenumber(value)
        if self.field_type == "bool":
            return self._validate_bool(value)
        if self._has_options():
            return self._validate_select(value, self.options)
        print(f"WARNING: No validator for field type {self.field_type}: {value}")
        return None

    def _validation_error(self, expected: str, value: Any):
        print(
            f"WARNING: Invalid format for field {self.name} "
            f"expected {expected} (type={self.field_type}) given {value} (type={type(value)})"
        )

    # In JavaScript, date is actually a timestamp which ideally should be human-readable and ISO 8601
    def _validate_date(self, value: Any):
        if value is None:
            return None

        if isinstance(value, datetime.datetime):
            dt_value = value
            if dt_value.tzinfo is None:
                dt_value = dt_value.replace(tzinfo=get_local_timezone())
            return dt_value

        if not isinstance(value, str):
            print(f"WARNING: Unrecognized date type {type(value)}: {value}")
            return None

        try:
            parsed_date = parser.parse(str(value))

            if (
                parsed_date.tzinfo is None
                or parsed_date.tzinfo.utcoffset(parsed_date) is None
            ):
                parsed_date = pytz.UTC.localize(parsed_date)  # noqa

                pst = pytz.timezone("America/Los_Angeles")
                parsed_date = pst.localize(
                    datetime.datetime.combine(
                        parsed_date.date(), datetime.time(hour=13)
                    )
                )
                parsed_date = parsed_date.astimezone(pytz.UTC)

            return parsed_date

        except (TypeError, ValueError) as e:
            print(f"WARNING parsing date: {e}")
            self._validation_error("timestamp", value)
            return datetime.datetime.now(pytz.UTC)

    def _validate_html(self, value: Any):
        return self._validate_text(value)

    def _validate_number(self, value: Any):
        try:
            return float(value)
        except ValueError:
            self._validation_error("float", value)
            return None

    def _validate_phonenumber(self, value: Any, default_region="US"):
        try:
            parsed_number = phonenumbers.parse(value, default_region)
            if phonenumbers.is_valid_number(parsed_number):
                return phonenumbers.format_number(
                    parsed_number, phonenumbers.PhoneNumberFormat.E164
                )
            else:
                self._validation_error("invalid", value)
        except phonenumbers.phonenumberutil.NumberParseException:
            self._validation_error("cannot parse", value)

        return None

    def _validate_bool(self, value: Any):
        try:
            return bool(value)
        except ValueError:
            self._validation_error("bool", value)
            return False

    def _validate_select(self, value: Any, options: List[Option]):
        option_values = [option.value for option in options]
        if isinstance(value, str):
            if value in option_values:
                return value
            # Sometimes GPT outputs the Label instead of the Value
            option_labels = {option.label: option.value for option in options}
            if value in option_labels:
                return option_labels[value]

            # Do one more try with "fuzzy match"
            print(
                f"WARNING: GPT response with {value} which ain't a value nor label for field {self.name}"
            )
            for option in options:
                if value in f"{option.value}: {option.label}":  # noqa: E701
                    return option.value

            self._validation_error("str value not an option label or value ", value)

        # Sometimes we get `'hs_task_type': {'label': 'Call', 'value': 'CALL'}`
        if isinstance(value, dict):
            if "value" in value:
                return self._validate_select(value["value"], options)

        # And sometimes we get `'hs_task_status': ['Not Started', 'NOT_STARTED']`
        if isinstance(value, (list, tuple)):
            if len(value) == 2:
                return self._validate_select(value[1], options)
            print(
                f"WARNING: Un-expected number of list items for an options field: {value}"
            )
            return self._validate_select(value[0], options)

        self._validation_error("unexpected format", value)
        return None

    def _validate_text(self, value: Any):
        if isinstance(value, str) or value is None:
            return value

        if isinstance(value, list):
            return "\n".join(f"* {item}" for item in value)

        if isinstance(value, dict):
            return "\n".join(f"* {key}: {value}" for key, value in value.items())

        self._validation_error("unexpected text type", value)
        return str(value)


# Mostly used for code-gen
class FormDefinition:
    def __init__(self, form_name: str, fields: List[FieldDefinition]):
        # Idea: Add a GPT explanation of the form, so we can classify the voice-memo.
        self.form_name = form_name
        self.fields = fields

    def get_field(self, field_name):
        for field in self.fields:
            if field.name == field_name:
                return field

        # This function is oftentimes used to check if name is in the field list so only warning.
        # It's a bit annoying, but can be lifesaving when developing.
        print(f"WARNING: Requested field {self.form_name}.{field_name} not in list")
        return None

    def get_field_names(self) -> list[str]:
        return [field.name for field in self.fields]

    def to_python_definition(self):
        return ",\n".join([field.to_python_definition() for field in self.fields])


# Related to HubspotObject
class FormData:
    def __init__(
        self,
        form: FormDefinition,
        data: Optional[dict] = None,
        omit_unknown_fields: bool = False,
    ):
        self.form = form
        self.data = {}
        if data is None:
            data = {}

        for field_name, value in data.items():
            self.set_field_value(
                field_name, value, raise_key_error=not omit_unknown_fields
            )

        # Fill in the defaults
        for field in form.fields:
            if field.name not in data or data[field.name] is None:
                if field.default_value is not None:
                    print(
                        f"Filling in default value for {form.form_name}.{field.name} to {field.default_value}"
                    )
                    self.set_field_value(field.name, field.default_value)

    def get_field(self, field_name) -> FieldDefinition:
        return self.form.get_field(field_name)

    def get_value(self, field_name: str, default_value=None):
        return self.data[field_name] if field_name in self.data else default_value

    def get_display_value(self, field_name: str) -> str:
        field = self.get_field(field_name)
        if bool(field):
            return field.display_value(self.data.get(field_name))
        return "None"

    def set_field_value(self, field_name: str, value: Any, raise_key_error=False):
        field = self.get_field(field_name)
        if bool(field):
            self.data[field_name] = field.validate_and_fix(value)
        else:
            error = (
                f"Field '{self.form.form_name}.{field_name}' "
                f"does not exist in FormDefinition {self.form.get_field_names()}"  # noqa: E713
            )
            if raise_key_error:
                raise KeyError(error)
            else:
                print(f"WARNING: Skipping {error}")

    def to_display_tuples(self) -> List[Tuple[str, str]]:
        result = []
        for field in self.form.fields:
            if field.ignore_in_display:
                print(f"INFO: ignoring {field.name} for to_display_tuples")
                continue
            result.append((field.label, self.get_display_value(field.name)))

        return result

    # TODO: We maybe want ordered dict.
    def to_dict(self) -> dict:
        return self.data.copy()
